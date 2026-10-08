from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials

from odds_api import (
    OddsApi,
    FD_TO_ODDS_SPORT,
    best_h2h_prices,
    named_h2h_prices,
    named_totals_prices,
    all_totals_prices,
    devig_three_way,
    match_fixture,
    PREFERRED_BOOKMAKERS,
)
from bet_tracking import (
    combo_probs,
    COMBO_SPECS,
    evaluate_bet,
    decimal_to_fractional,
)
from performance import apply_performance_filter


# ================= CONFIG =================
BASE_COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",          # Big 5
    "CL", "EL", "EC",                        # European cups
    "ELC", "EL1", "EL2",                     # Championship, League One, League Two
    "NL",                                    # English National League
    "SPL",                                   # Scottish Premiership
    "SD",                                    # Spanish Segunda División (2nd)
    "SF",                                    # Spanish Primera Federación (3rd)
    "DED", "PPL",                            # Eredivisie + Primeira Liga
]

OPTIONAL_COMP_CODES = [
    "BSA",   # Brazil Serie A
    "MLS",   # Major League Soccer
]

SLEEP_SECONDS = 7.0
DAYS_AHEAD = 3
HISTORY_DAYS = 210
MAX_GOALS = 10

TOP_N = 10
TOP_MIX = 20

UNIQUE_TEAMS_PER_TOP10 = True
UNIQUE_TEAMS_IN_TOP20 = True

# Recency weighting
RECENT_N = 10
RECENT_WEIGHT = 1.8

# Bayesian shrinkage
SHRINK_K = 8.0

# Head-to-head (optional)
USE_H2H = True
H2H_MATCHES_LOOKBACK = 6
H2H_SHRINK_K = 4.0
H2H_MAX_GOAL_ADJ = 0.15
H2H_BLEND = 0.35

# Confidence
CONF_K = 12.0

# Market odds blend (model probability vs. de-vigged bookmaker probability)
MARKET_BLEND_ALPHA = 0.5

# ---- ODDS FILTER ----
MIN_DECIMAL_ODDS = 1.20
ODDS_FILTER_REQUIRE_NAMED_BOOK = False

# ---- VALUE / EDGE FILTER ----
# Only keep a pick if model probability beats the bookmaker's implied
# probability by at least this amount (e.g. 0.04 = 4% edge)
MIN_EDGE = 0.04

# ---- PERFORMANCE FILTER ----
PERF_MIN_WIN_RATE = 0.52
PERF_MIN_BETS = 8

# Combo settings
COMBO_MIN_PROB = 0.35
COMBO_TOP_N = 25

# Tabs
TAB_FIXTURES = "Fixtures"
TAB_TEAM_FORM = "Team_Form"
TAB_PICKS = "Picks"
TAB_TOP20 = "Top20_Mix"
TAB_ACCESS = "Competitions_Access"
TAB_SAFE = "Safe_Picks"
TAB_BAL = "Balanced_Picks"
TAB_BEST_BETS = "Best_Bets"
TAB_COMBOS = "Combo_Bets"
TAB_HISTORY = "Bet_History"
TAB_ACCURACY = "Accuracy"


# ================= UTILS =================
def now() -> datetime:
    return datetime.now(timezone.utc)


def log(msg: str) -> None:
    print(msg, flush=True)


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ================= SMART SCORE V2 =================
def market_weight(bet: str) -> float:
    b = bet.upper()
    if b in ("HOME WIN", "AWAY WIN"):
        return 1.00
    if b == "OVER 1.5":                     # boosted – best performer
        return 1.08
    if b == "DRAW":
        return 0.88
    if b.startswith("DOUBLE CHANCE"):
        return 0.92
    if b in ("BTTS YES", "BTTS NO"):
        return 0.94
    if b.startswith("OVER ") or b.startswith("UNDER "):
        return 0.92
    if "CLEAN SHEET" in b or "WIN TO NIL" in b:
        return 0.86
    if "&" in b:
        return 0.88
    if "DNB" in b:
        return 0.93
    return 0.90


def smart_score_v2(
    prob: float,
    confidence: float,
    bet: str,
    home_xg: float,
    away_xg: float,
    league_base_total: float,
) -> Tuple[float, float, float]:
    p = clamp(float(prob), 0.0, 1.0)
    c = clamp(float(confidence), 0.0, 1.0)

    tot = max(0.1, float(home_xg) + float(away_xg))
    base_tot = max(0.1, float(league_base_total))
    gd = float(home_xg) - float(away_xg)

    extremeness = abs(p - 0.5) * 2.0
    penalty = (extremeness ** 1.25) * (1.0 - c) * 0.28
    penalty = clamp(penalty, 0.0, 0.22)

    adj = 1.0
    b = bet.upper()

    if b == "DRAW":
        draw_suppress = 1.0 - clamp((tot / base_tot - 1.0) * 0.18, 0.0, 0.18)
        draw_suppress *= (1.0 - clamp(abs(gd) * 0.10, 0.0, 0.20))
        adj *= draw_suppress
    elif b == "HOME WIN":
        adj *= (1.0 + clamp(gd * 0.08, -0.12, 0.18))
    elif b == "AWAY WIN":
        adj *= (1.0 + clamp((-gd) * 0.08, -0.12, 0.18))
    elif b.startswith("OVER "):
        adj *= (1.0 + clamp((tot / base_tot - 1.0) * 0.10, -0.08, 0.10))
    elif b.startswith("UNDER "):
        adj *= (1.0 + clamp((1.0 - tot / base_tot) * 0.10, -0.08, 0.10))
    elif b in ("BTTS YES", "BTTS NO"):
        bal = 1.0 - clamp(abs(gd) * 0.10, 0.0, 0.12)
        if b == "BTTS YES":
            tot_adj = 1.0 + clamp((tot / base_tot - 1.0) * 0.06, -0.06, 0.06)
        else:
            tot_adj = 1.0 + clamp((1.0 - tot / base_tot) * 0.06, -0.06, 0.06)
        adj *= bal * tot_adj

    adj = clamp(adj, 0.80, 1.18)
    w = market_weight(bet)
    score = (p * c) * w * adj * (1.0 - penalty)
    return float(score), float(penalty), float(adj)


# ================= GOAL MODEL =================
def match_probs(lh: float, la: float) -> Dict[str, float]:
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = 0.0
    p_btts_yes = 0.0
    p_over_0_5 = 0.0
    p_over_1_5 = 0.0
    p_over_2_5 = 0.0
    p_over_3_5 = 0.0

    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

            if i > 0 and j > 0:
                p_btts_yes += p

            tg = i + j
            if tg > 0:
                p_over_0_5 += p
            if tg > 1:
                p_over_1_5 += p
            if tg > 2:
                p_over_2_5 += p
            if tg > 3:
                p_over_3_5 += p

    p_btts_no = 1.0 - p_btts_yes
    p_1x = p_home + p_draw
    p_x2 = p_draw + p_away
    p_12 = p_home + p_away

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_1x": p_1x, "p_x2": p_x2, "p_12": p_12,
        "p_btts_yes": p_btts_yes, "p_btts_no": p_btts_no,
        "p_over_0_5": p_over_0_5, "p_over_1_5": p_over_1_5,
        "p_over_2_5": p_over_2_5, "p_over_3_5": p_over_3_5,
        "p_under_1_5": 1.0 - p_over_1_5,
        "p_under_2_5": 1.0 - p_over_2_5,
        "p_under_3_5": 1.0 - p_over_3_5,
    }


# ================= GOOGLE SHEETS =================
def gs() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        json.loads(env("GOOGLE_SERVICE_ACCOUNT_JSON")),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)


def open_sheet():
    return gs().open_by_key(env("SHEET_ID"))


def _get_or_create_ws(sh: gspread.Spreadsheet, title: str, rows: int = 4000, cols: int = 60) -> gspread.Worksheet:
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title, rows, cols)


def write_df(name: str, df: pd.DataFrame):
    sh = open_sheet()
    ws = _get_or_create_ws(sh, name)
    ws.clear()
    if df is None or df.empty:
        ws.update([["(no data)"]])
    else:
        ws.update(
            [list(df.columns)] + df.fillna("").astype(str).values.tolist(),
            value_input_option="USER_ENTERED",
        )


def read_df(name: str) -> pd.DataFrame:
    sh = open_sheet()
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()
    vals = ws.get_all_values()
    if len(vals) < 2:
        return pd.DataFrame()
    header = vals[0]
    if len(header) == 1 and header[0].strip().lower().startswith("(no data"):
        return pd.DataFrame()
    return pd.DataFrame(vals[1:], columns=header)


def sort_by_date(df: pd.DataFrame, date_col: str = "utcDate") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out["_sort_dt"] = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    out = out.sort_values("_sort_dt", ascending=True, na_position="last").drop(columns=["_sort_dt"])
    return out.reset_index(drop=True)


def set_visible_tabs(keep_titles: List[str]):
    sh = open_sheet()
    meta = sh.fetch_sheet_metadata()
    reqs = []
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        title = props.get("title")
        sid = props.get("sheetId")
        if title is None or sid is None:
            continue
        hidden = title not in keep_titles
        reqs.append({
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "hidden": hidden},
                "fields": "hidden",
            }
        })
    if reqs:
        sh.batch_update({"requests": reqs})


# ================= FOOTBALL-DATA API =================
@dataclass
class FD:
    token: str
    base: str = "https://api.football-data.org/v4"

    def get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        r = requests.get(
            self.base + path,
            headers={"X-Auth-Token": self.token},
            params=params or {},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def matches(self, code: str, status: str, d1: str, d2: str) -> List[Dict[str, Any]]:
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": d1, "dateTo": d2},
        ).get("matches", [])

    def competitions(self) -> List[Dict[str, Any]]:
        return self.get("/competitions", {}).get("competitions", [])


# ================= MODEL CORE =================
def recency_weighted_mean(values: List[float], recent_n: int = RECENT_N, recent_weight: float = RECENT_WEIGHT) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    n = len(vals)
    weights = [1.0] * n
    for i in range(max(0, n - recent_n), n):
        weights[i] = recent_weight
    wsum = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / wsum


def shrink(mean_est: float, n: float, prior_mean: float, k: float = SHRINK_K) -> float:
    if n <= 0 or math.isnan(mean_est):
        return prior_mean
    return (n * mean_est + k * prior_mean) / (n + k)


def compute_league_baselines(rs: pd.DataFrame) -> pd.DataFrame:
    g = rs.groupby("league")
    return pd.DataFrame({"home_gf": g["hg"].mean(), "away_gf": g["ag"].mean(), "n": g.size()})


def compute_team_indices(rs: pd.DataFrame, league_baselines: pd.DataFrame) -> pd.DataFrame:
    rs = rs.sort_values("utcDate")
    rows = []
    for league, sub in rs.groupby("league"):
        base_home = safe_float(league_baselines.loc[league, "home_gf"], 1.35) if league in league_baselines.index else 1.35
        base_away = safe_float(league_baselines.loc[league, "away_gf"], 1.10) if league in league_baselines.index else 1.10

        for team, tsub in sub.groupby("home"):
            hg = tsub["hg"].tolist()
            ag = tsub["ag"].tolist()
            n = len(hg)
            rows.append({
                "league": league, "team": team,
                "home_hg": shrink(recency_weighted_mean(hg), n, base_home),
                "home_ag": shrink(recency_weighted_mean(ag), n, base_away),
                "n_home": n
            })

        for team, tsub in sub.groupby("away"):
            ag = tsub["ag"].tolist()
            hg = tsub["hg"].tolist()
            n = len(ag)
            rows.append({
                "league": league, "team": team,
                "away_hg": shrink(recency_weighted_mean(ag), n, base_away),
                "away_ag": shrink(recency_weighted_mean(hg), n, base_home),
                "n_away": n
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["league", "team"], as_index=False).agg({
        "home_hg": "max", "home_ag": "max",
        "away_hg": "max", "away_ag": "max",
        "n_home": "max", "n_away": "max",
    }).fillna(0.0)

    out = []
    for _, r in agg.iterrows():
        league = r["league"]
        base_home = safe_float(league_baselines.loc[league, "home_gf"], 1.35) if league in league_baselines.index else 1.35
        base_away = safe_float(league_baselines.loc[league, "away_gf"], 1.10) if league in league_baselines.index else 1.10

        home_attack = (r["home_hg"] / base_home) if base_home > 0 else 1.0
        home_def = (r["home_ag"] / base_away) if base_away > 0 else 1.0
        away_attack = (r["away_hg"] / base_away) if base_away > 0 else 1.0
        away_def = (r["away_ag"] / base_home) if base_home > 0 else 1.0

        out.append({
            "league": league,
            "team": r["team"],
            "home_attack": clamp(float(home_attack), 0.55, 1.75),
            "home_defense": clamp(float(home_def), 0.55, 1.75),
            "away_attack": clamp(float(away_attack), 0.55, 1.75),
            "away_defense": clamp(float(away_def), 0.55, 1.75),
            "n_home": float(r["n_home"]),
            "n_away": float(r["n_away"]),
        })

    return pd.DataFrame(out).set_index(["league", "team"])


def h2h_goal_adjustment(rs: pd.DataFrame, league: str, home: str, away: str) -> Tuple[float, float]:
    if not USE_H2H or rs.empty:
        return 0.0, 0.0

    sub = rs[(rs["league"] == league) & (
        ((rs["home"] == home) & (rs["away"] == away)) |
        ((rs["home"] == away) & (rs["away"] == home))
    )].sort_values("utcDate")

    if sub.empty:
        return 0.0, 0.0

    sub = sub.tail(H2H_MATCHES_LOOKBACK)

    hg, ag = [], []
    for _, r in sub.iterrows():
        if r["home"] == home:
            hg.append(float(r["hg"]))
            ag.append(float(r["ag"]))
        else:
            hg.append(float(r["ag"]))
            ag.append(float(r["hg"]))

    n = len(hg)
    if n == 0:
        return 0.0, 0.0

    hmean = recency_weighted_mean(hg, min(RECENT_N, n), RECENT_WEIGHT)
    amean = recency_weighted_mean(ag, min(RECENT_N, n), RECENT_WEIGHT)

    mean_total = (hmean + amean) / 2.0
    dh = (hmean - mean_total)
    da = (amean - mean_total)

    shrink_factor = n / (n + H2H_SHRINK_K)
    dh *= shrink_factor * H2H_BLEND
    da *= shrink_factor * H2H_BLEND

    return clamp(dh, -H2H_MAX_GOAL_ADJ, H2H_MAX_GOAL_ADJ), clamp(da, -H2H_MAX_GOAL_ADJ, H2H_MAX_GOAL_ADJ)


def confidence_score(n_home: float, n_away: float) -> float:
    n = max(0.0, float(n_home or 0.0) + float(n_away or 0.0))
    return float(n / (n + CONF_K))


# ================= MARKET ODDS INTEGRATION =================
def fetch_market_odds(fx_df: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    api = OddsApi(env("ODDS_API_KEY"))
    events_by_sport: Dict[str, List[Dict[str, Any]]] = {}
    matched: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for league, sub in fx_df.groupby("league"):
        sport_key = FD_TO_ODDS_SPORT.get(league)
        if not sport_key:
            continue
        if sport_key not in events_by_sport:
            try:
                events_by_sport[sport_key] = api.get_odds(sport_key, markets=["h2h", "totals"])
            except Exception as e:
                log(f"odds fetch failed for {sport_key}: {e}")
                events_by_sport[sport_key] = []

        for _, r in sub.iterrows():
            ev = match_fixture(r["home"], r["away"], r["utcDate"], events_by_sport[sport_key])
            if ev is None:
                continue

            named_h2h, book_h2h = named_h2h_prices(ev)
            named_tot, book_tot = named_totals_prices(ev)

            matched[(league, r["home"], r["away"])] = {
                "h2h": best_h2h_prices(ev),
                "named_h2h": named_h2h,
                "named_totals": named_tot,
                "all_totals": all_totals_prices(ev),
                "book": book_h2h or book_tot,
                "event": ev,
            }

    named_count = sum(1 for v in matched.values() if v.get("book"))
    log(f"odds matched: {len(matched)} fixtures, of which {named_count} have "
        f"{'/'.join(PREFERRED_BOOKMAKERS)} prices")
    return matched


def market_probs_for_fixture(odds_entry: Optional[Dict[str, Any]], home: str, away: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not odds_entry:
        return out
    h2h = odds_entry.get("h2h", {})
    if home in h2h and away in h2h and "Draw" in h2h:
        ph, pdw, pa = devig_three_way(h2h[home], h2h["Draw"], h2h[away])
        out["p_home"], out["p_draw"], out["p_away"] = ph, pdw, pa
    return out


def blend(model_p: float, market_p: Optional[float], alpha: float = MARKET_BLEND_ALPHA) -> float:
    if market_p is None:
        return model_p
    return alpha * model_p + (1.0 - alpha) * market_p


def odds_for_bet(
    odds_entry: Optional[Dict[str, Any]],
    bet: str,
    home: str,
    away: str,
) -> Tuple[Optional[float], str]:
    if not odds_entry:
        return None, "none"

    b = bet.strip().upper()
    named_h2h = odds_entry.get("named_h2h", {})
    named_tot = odds_entry.get("named_totals", {})
    all_h2h = odds_entry.get("h2h", {})
    all_tot = odds_entry.get("all_totals", {})
    book = odds_entry.get("book")

    if b in ("HOME WIN", "AWAY WIN", "DRAW"):
        key = home if b == "HOME WIN" else (away if b == "AWAY WIN" else "Draw")
        if key in named_h2h:
            return named_h2h[key], book or "named"
        if key in all_h2h:
            return all_h2h[key], "market-best"
        return None, "none"

    if b.startswith("OVER ") or b.startswith("UNDER "):
        try:
            line = float(b.split()[1])
        except (IndexError, ValueError):
            return None, "none"
        side = "Over" if b.startswith("OVER") else "Under"
        if line in named_tot and side in named_tot[line]:
            return named_tot[line][side], book or "named"
        if line in all_tot and side in all_tot[line]:
            return all_tot[line][side], "market-best"
        return None, "none"

    return None, "none"


def passes_odds_filter(price: Optional[float], source: str) -> bool:
    if price is None:
        return not ODDS_FILTER_REQUIRE_NAMED_BOOK
    return float(price) >= MIN_DECIMAL_ODDS


def attach_odds(df: pd.DataFrame, odds_map: Dict[Tuple[str, str, str], Dict[str, Any]],
                apply_filter: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    prices, fracs, sources, keeps = [], [], [], []
    for _, r in df.iterrows():
        entry = odds_map.get((r.get("league"), r.get("home"), r.get("away")))
        price, source = odds_for_bet(entry, str(r.get("bet", "")), r.get("home"), r.get("away"))
        prices.append(round(price, 3) if price else "")
        fracs.append(decimal_to_fractional(price) if price else "")
        sources.append(source)
        keeps.append(passes_odds_filter(price, source))

    out = df.copy()
    out["odds"] = prices
    out["fractional"] = fracs
    out["odds_source"] = sources

    if apply_filter:
        before = len(out)
        out = out[pd.Series(keeps, index=out.index)].reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            log(f"  odds filter removed {dropped} pick(s) priced shorter than "
                f"{MIN_DECIMAL_ODDS} ({decimal_to_fractional(MIN_DECIMAL_ODDS)})")
        if "rank" in out.columns and not out.empty:
            out["rank"] = range(1, len(out) + 1)
    return out


def apply_value_filter(df: pd.DataFrame, min_edge: float = MIN_EDGE) -> pd.DataFrame:
    """
    Keep only picks that have real odds AND where the model probability
    beats the bookmaker implied probability by at least min_edge.
    """
    if df is None or df.empty:
        return df

    if "odds" not in df.columns or "prob" not in df.columns:
        return df

    out = df.copy()
    out["_odds_num"] = pd.to_numeric(out["odds"], errors="coerce")
    out["_prob_num"] = pd.to_numeric(out["prob"], errors="coerce")

    # implied probability = 1 / decimal odds
    out["_implied"] = 1.0 / out["_odds_num"]
    out["_edge"] = out["_prob_num"] - out["_implied"]

    before = len(out)
    out = out[out["_edge"] >= min_edge].copy()
    dropped = before - len(out)
    if dropped:
        log(f"  value filter removed {dropped} pick(s) with edge < {min_edge:.0%}")

    out = out.drop(columns=["_odds_num", "_prob_num", "_implied", "_edge"], errors="ignore")

    if "rank" in 
