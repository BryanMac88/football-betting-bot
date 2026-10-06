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


# ================= CONFIG =================
BASE_COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",          # Big 5
    "CL", "EL", "EC",                        # European cups
    "ELC", "EL1", "EL2",                     # English Championship, L1, L2
    "NL",                                    # English National League
    "SPL",                                   # Scottish Premiership
    "SD",                                    # Spanish Segunda (2nd)
    "SF",                                    # Spanish Primera Federación (3rd)
    "DED", "PPL",                            # Eredivisie + Primeira
]

OPTIONAL_COMP_CODES = [
    "BSA",  # Brazil Serie A
    "MLS",  # Major League Soccer
    "BJL",  # Belgian Pro League (if available)
    "TSL",  # Turkish Super Lig (if available)
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
# Exclude odds-on favourites priced SHORTER than 1/5 fractional.
# 1/5 fractional == 1.20 decimal. Anything shorter (1/6 = 1.1667,
# 1/7 = 1.1429, 1/10 = 1.10) is excluded: tiny payout for the risk.
MIN_DECIMAL_ODDS = 1.20

# When neither Paddy Power nor Boylesports covers a fixture, we have no
# verified price to filter on. True  = drop the pick (strict).
#                                False = keep it, flagged odds_source="none".
# Start False so you can see how often it happens before tightening.
ODDS_FILTER_REQUIRE_NAMED_BOOK = False

# Minimum probability for a combined bet to be listed at all — combos
# are inherently lower-probability, so the usual thresholds don't apply.
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
        # USER_ENTERED so Sheets recalculates dependent formulas after the
        # write, rather than leaving them stale until manually nudged.
        ws.update(
            [list(df.columns)] + df.fillna("").astype(str).values.tolist(),
            value_input_option="USER_ENTERED",
        )


def read_df(name: str) -> pd.DataFrame:
    """Read an existing tab back into a DataFrame. Returns an empty frame
    if the tab doesn't exist yet or only holds the '(no data)' placeholder."""
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
    """Fetch Odds API events per league and fuzzy-match each fixture.
    Now pulls BOTH h2h and totals so the odds filter can price
    over/under picks too, and records which named bookmaker (if any)
    the price came from."""
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
                "h2h": best_h2h_prices(ev),                # for probability blending
                "named_h2h": named_h2h,                     # for the odds filter
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
    """Look up the decimal price for a given bet label.
    Returns (price, source) where source is the bookmaker key,
    'market-best' for a best-across-books fallback, or 'none'.

    Combined bets and double-chance/BTTS aren't in the standard Odds API
    market set, so they return (None, 'none') — they're listed with the
    model's own implied price instead, clearly labelled."""
    if not odds_entry:
        return None, "none"

    b = bet.strip().upper()
    named_h2h = odds_entry.get("named_h2h", {})
    named_tot = odds_entry.get("named_totals", {})
    all_h2h = odds_entry.get("h2h", {})
    all_tot = odds_entry.get("all_totals", {})
    book = odds_entry.get("book")

    # 1X2
    if b in ("HOME WIN", "AWAY WIN", "DRAW"):
        key = home if b == "HOME WIN" else (away if b == "AWAY WIN" else "Draw")
        if key in named_h2h:
            return named_h2h[key], book or "named"
        if key in all_h2h:
            return all_h2h[key], "market-best"
        return None, "none"

    # Over / Under totals
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
    """Exclude anything priced shorter than MIN_DECIMAL_ODDS (1/5).
    Unpriced picks are governed by ODDS_FILTER_REQUIRE_NAMED_BOOK."""
    if price is None:
        return not ODDS_FILTER_REQUIRE_NAMED_BOOK
    return float(price) >= MIN_DECIMAL_ODDS


def attach_odds(df: pd.DataFrame, odds_map: Dict[Tuple[str, str, str], Dict[str, Any]],
                apply_filter: bool = True) -> pd.DataFrame:
    """Add odds / fractional / odds_source columns to a picks frame, and
    optionally drop rows failing the minimum-odds rule."""
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


# ================= EDGE BASELINES =================
MARKET_COLS = {
    "HOME WIN": "p_home",
    "DRAW": "p_draw",
    "AWAY WIN": "p_away",
    "BTTS YES": "p_btts_yes",
    "BTTS NO": "p_btts_no",
    "OVER 1.5": "p_over_1_5",
    "OVER 2.5": "p_over_2_5",
    "UNDER 2.5": "p_under_2_5",
}

SAFE_RULES = {
    "min_conf": 0.60, "min_prob": 0.58,
    "no_bet_low": 0.45, "no_bet_high": 0.55,
    "edge_win": 0.05, "edge_goals": 0.08,
    "max_total_xg_over_base": 1.20,
}

BAL_RULES = {
    "min_conf": 0.50, "min_prob": 0.55,
    "no_bet_low": 0.44, "no_bet_high": 0.56,
    "edge_win": 0.04, "edge_goals": 0.06,
    "max_total_xg_over_base": 1.50,
}


def build_league_market_baselines(probs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for league, sub in probs_df.groupby("league"):
        for mkt, col in MARKET_COLS.items():
            if col in sub.columns:
                rows.append({"league": league, "market": mkt, "league_avg_prob": float(sub[col].mean())})
    return pd.DataFrame(rows)


def make_filtered_picks(
    probs_df: pd.DataFrame,
    baselines: pd.DataFrame,
    rules: Dict[str, float],
    max_rows: int = 30,
    unique_teams: bool = True
) -> pd.DataFrame:
    if probs_df.empty or baselines.empty:
        return pd.DataFrame()

    base_map = {(r["league"], r["market"]): float(r["league_avg_prob"]) for _, r in baselines.iterrows()}

    picks = []
    for _, r in probs_df.iterrows():
        league, home, away = r["league"], r["home"], r["away"]
        conf = float(r["confidence"])
        home_xg, away_xg = float(r["home_xg"]), float(r["away_xg"])
        base_tot = float(r["league_base_total"])
        tot_xg = home_xg + away_xg

        if tot_xg > base_tot + float(rules["max_total_xg_over_base"]):
            continue

        for mkt, col in MARKET_COLS.items():
            prob = float(r[col])
            if prob < float(rules["min_prob"]):
                continue
            if float(rules["no_bet_low"]) < prob < float(rules["no_bet_high"]):
                continue
            if conf < float(rules["min_conf"]):
                continue

            league_avg = base_map.get((league, mkt))
            if league_avg is None:
                continue
            edge = prob - league_avg

            if mkt in ("HOME WIN", "DRAW", "AWAY WIN"):
                if edge < float(rules["edge_win"]):
                    continue
            else:
                if edge < float(rules["edge_goals"]):
                    continue

            sc, pen, adj = smart_score_v2(prob, conf, mkt, home_xg, away_xg, base_tot)

            picks.append({
                "utcDate": r["utcDate"], "league": league, "home": home, "away": away,
                "bet": mkt, "prob": round(prob, 3),
                "league_avg": round(league_avg, 3), "edge": round(edge, 3),
                "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2),
                "confidence": round(conf, 3), "adj": round(adj, 3),
                "penalty": round(pen, 3), "score": round(sc, 4),
            })

    df = pd.DataFrame(picks)
    if df.empty:
        return df

    df = df.sort_values(["score", "edge", "prob"], ascending=[False, False, False]).reset_index(drop=True)

    if unique_teams:
        used = set()
        keep = []
        for _, row in df.iterrows():
            h, a = row["home"], row["away"]
            if h in used or a in used:
                continue
            keep.append(row)
            used.add(h)
            used.add(a)
            if len(keep) >= max_rows:
                break
        df = pd.DataFrame(keep)

    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ================= TOP10 / MIX HELPERS =================
CORE_COLS = ["utcDate", "league", "home", "away"]


def _dedupe_teams(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    used = set()
    kept = []
    for _, r in df.iterrows():
        h, a = r.get("home"), r.get("away")
        if h in used or a in used:
            continue
        kept.append(r)
        if h:
            used.add(h)
        if a:
            used.add(a)
        if len(kept) >= limit:
            break
    if not kept:
        return df.head(0)
    return pd.DataFrame(kept).reset_index(drop=True)


def top_n_for_market(probs_df: pd.DataFrame, prob_col: str, bet_label: str, top_n: int = TOP_N) -> pd.DataFrame:
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    base_cols = CORE_COLS + ["confidence", "home_xg", "away_xg", "league_base_total", prob_col]
    df = probs_df[base_cols].copy().rename(columns={prob_col: "prob"})
    df["bet"] = bet_label
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["prob"])

    scores = [
        smart_score_v2(float(r["prob"]), float(r["confidence"]), bet_label,
                       float(r["home_xg"]), float(r["away_xg"]), float(r["league_base_total"]))
        for _, r in df.iterrows()
    ]
    df["score"] = [s[0] for s in scores]
    df["penalty"] = [s[1] for s in scores]
    df["adj"] = [s[2] for s in scores]

    df = df.sort_values(["score", "prob", "utcDate"], ascending=[False, False, True]).reset_index(drop=True)
    df = _dedupe_teams(df, top_n) if UNIQUE_TEAMS_PER_TOP10 else df.head(top_n).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"]]


def build_top20_mix(probs_df: pd.DataFrame, top_k: int = TOP_MIX) -> pd.DataFrame:
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    def _all_for(col: str, label: str) -> pd.DataFrame:
        d = probs_df[CORE_COLS + ["confidence", "home_xg", "away_xg", "league_base_total", col]].copy()
        d = d.rename(columns={col: "prob"})
        d["bet"] = label
        d["prob"] = pd.to_numeric(d["prob"], errors="coerce")
        d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
        d = d.dropna(subset=["prob"])
        scores = [
            smart_score_v2(float(r["prob"]), float(r["confidence"]), label,
                           float(r["home_xg"]), float(r["away_xg"]), float(r["league_base_total"]))
            for _, r in d.iterrows()
        ]
        d["score"] = [s[0] for s in scores]
        d["penalty"] = [s[1] for s in scores]
        d["adj"] = [s[2] for s in scores]
        return d[["utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"]]

    parts = [
        _all_for("p_home", "HOME WIN"), _all_for("p_draw", "DRAW"), _all_for("p_away", "AWAY WIN"),
        _all_for("p_btts_yes", "BTTS YES"), _all_for("p_btts_no", "BTTS NO"),
        _all_for("p_over_1_5", "OVER 1.5"), _all_for("p_over_2_5", "OVER 2.5"),
        _all_for("p_over_3_5", "OVER 3.5"), _all_for("p_under_2_5", "UNDER 2.5"),
        _all_for("p_1x", "DOUBLE CHANCE 1X"), _all_for("p_x2", "DOUBLE CHANCE X2"),
        _all_for("p_12", "DOUBLE CHANCE 12"),
    ]

    mix = pd.concat(parts, ignore_index=True)
    mix = mix.drop_duplicates(subset=["utcDate", "home", "away", "bet"])
    mix = mix.sort_values(["score", "prob", "utcDate"], ascending=[False, False, True]).reset_index(drop=True)
    mix = _dedupe_teams(mix, top_k) if UNIQUE_TEAMS_IN_TOP20 else mix.head(top_k).reset_index(drop=True)
    mix.insert(0, "rank", range(1, len(mix) + 1))
    return mix[["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"]]


# ================= BEST BETS =================
BEST_BET_MARKETS = [
    ("p_home", "HOME WIN"), ("p_draw", "DRAW"), ("p_away", "AWAY WIN"),
    ("p_btts_yes", "BTTS YES"), ("p_btts_no", "BTTS NO"),
    ("p_over_1_5", "OVER 1.5"), ("p_over_2_5", "OVER 2.5"), ("p_under_2_5", "UNDER 2.5"),
    ("p_1x", "DOUBLE CHANCE 1X"), ("p_x2", "DOUBLE CHANCE X2"),
]


def build_best_bets(probs_df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: its single highest-probability outcome across
    all markets, ranked by that probability descending."""
    if probs_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in probs_df.iterrows():
        best_mkt, best_p = None, -1.0
        for col, label in BEST_BET_MARKETS:
            p = float(r.get(col, 0.0))
            if p > best_p:
                best_p, best_mkt = p, label
        rows.append({
            "utcDate": r["utcDate"], "league": r["league"],
            "home": r["home"], "away": r["away"],
            "bet": best_mkt, "prob": round(best_p, 3),
            "confidence": r["confidence"],
            "has_market_odds": bool(r.get("has_market_odds", False)),
        })

    df = pd.DataFrame(rows).sort_values(["prob", "confidence"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ================= COMBO BETS =================
def build_combo_bets(probs_df: pd.DataFrame, top_k: int = COMBO_TOP_N) -> pd.DataFrame:
    """Combined ("&") bets, e.g. AWAY WIN & OVER 2.5.

    NOTE: these probabilities are MODEL-ONLY. The 1X2 probabilities
    elsewhere get blended with de-vigged market odds, but a joint
    probability can't be blended the same way without joint market
    prices, which the Odds API doesn't supply for these markets. So
    treat combo numbers as the model's own view, not a market-corrected
    one — that's also why no bookmaker odds column appears here."""
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in probs_df.iterrows():
        lh, la = float(r["home_xg"]), float(r["away_xg"])
        conf = float(r["confidence"])
        base_tot = float(r["league_base_total"])
        cp = combo_probs(lh, la, MAX_GOALS)

        for label, prob in cp.items():
            if prob < COMBO_MIN_PROB:
                continue
            sc, pen, adj = smart_score_v2(prob, conf, label, lh, la, base_tot)
            rows.append({
                "utcDate": r["utcDate"], "league": r["league"],
                "home": r["home"], "away": r["away"],
                "bet": label, "prob": round(prob, 3),
                "model_implied_odds": round(1.0 / prob, 2) if prob > 0 else "",
                "confidence": round(conf, 3),
                "home_xg": round(lh, 2), "away_xg": round(la, 2),
                "score": round(sc, 4),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["score", "prob"], ascending=[False, False]).reset_index(drop=True)
    df = df.head(top_k).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ================= BET HISTORY / TRACKING =================
HISTORY_COLS = [
    "logged_utc", "utcDate", "league", "home", "away", "bet", "source",
    "prob", "odds", "fractional", "odds_source", "result", "score", "profit",
]


def _hist_key(row: Any) -> Tuple[str, str, str, str, str]:
    """Identity of a logged bet — one row per (fixture, bet, source)."""
    return (
        str(row.get("utcDate", ""))[:10],
        str(row.get("home", "")),
        str(row.get("away", "")),
        str(row.get("bet", "")),
        str(row.get("source", "")),
    )


def append_new_picks(history: pd.DataFrame, picks: pd.DataFrame, source: str) -> pd.DataFrame:
    """Append picks not already logged. Never rewrites existing rows —
    the track record must stay immutable, or a later model change would
    silently rewrite past predictions and make the accuracy stats
    meaningless."""
    if picks is None or picks.empty:
        return history

    existing = set()
    if not history.empty:
        for _, h in history.iterrows():
            existing.add(_hist_key(h))

    new_rows = []
    stamp = now().strftime("%Y-%m-%d %H:%M:%S UTC")
    for _, p in picks.iterrows():
        cand = {
            "utcDate": p.get("utcDate", ""), "home": p.get("home", ""),
            "away": p.get("away", ""), "bet": p.get("bet", ""), "source": source,
        }
        if _hist_key(cand) in existing:
            continue
        new_rows.append({
            "logged_utc": stamp,
            "utcDate": p.get("utcDate", ""),
            "league": p.get("league", ""),
            "home": p.get("home", ""),
            "away": p.get("away", ""),
            "bet": p.get("bet", ""),
            "source": source,
            "prob": p.get("prob", ""),
            "odds": p.get("odds", ""),
            "fractional": p.get("fractional", ""),
            "odds_source": p.get("odds_source", ""),
            "result": "PENDING",
            "score": "",
            "profit": "",
        })

    if not new_rows:
        return history

    log(f"  logging {len(new_rows)} new pick(s) from {source}")
    add = pd.DataFrame(new_rows)
    return add if history.empty else pd.concat([history, add], ignore_index=True)


def settle_history(history: pd.DataFrame, rs_df: pd.DataFrame) -> pd.DataFrame:
    """Fill in results for PENDING bets whose match has now finished."""
    if history is None or history.empty or rs_df is None or rs_df.empty:
        return history

    # (date, home, away) -> (hg, ag)
    results: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for _, r in rs_df.iterrows():
        key = (str(r["utcDate"])[:10], str(r["home"]), str(r["away"]))
        try:
            results[key] = (int(r["hg"]), int(r["ag"]))
        except (TypeError, ValueError):
            continue

    settled = 0
    out = history.copy()
    # Values read back from Sheets arrive as strings, so these columns get a
    # string dtype that rejects numeric assignment. Cast to object first.
    for col in ("result", "score", "profit"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(object)

    for i, h in out.iterrows():
        if str(h.get("result", "")).upper() not in ("", "PENDING"):
            continue
        key = (str(h.get("utcDate", ""))[:10], str(h.get("home", "")), str(h.get("away", "")))
        if key not in results:
            continue
        hg, ag = results[key]
        won = evaluate_bet(str(h.get("bet", "")), hg, ag)
        if won is None:
            continue

        out.at[i, "result"] = "WON" if won else "LOST"
        out.at[i, "score"] = f"{hg}-{ag}"

        try:
            odds = float(h.get("odds", "") or 0)
        except (TypeError, ValueError):
            odds = 0.0
        if odds > 1:
            out.at[i, "profit"] = round(odds - 1.0, 3) if won else -1.0
        else:
            out.at[i, "profit"] = ""  # no verified price, can't compute ROI
        settled += 1

    if settled:
        log(f"  settled {settled} previously-pending bet(s)")
    return out


def _odds_bucket(odds: Any) -> str:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return "no price"
    if o < 1.5:
        return "1.20-1.49"
    if o < 2.0:
        return "1.50-1.99"
    if o < 3.0:
        return "2.00-2.99"
    if o < 5.0:
        return "3.00-4.99"
    return "5.00+"


def build_accuracy(history: pd.DataFrame) -> pd.DataFrame:
    """Hit rate and ROI per bet type, and per odds band — the whole point
    of tracking: showing which kinds of pick actually perform, rather
    than which ones the model merely felt confident about."""
    if history is None or history.empty:
        return pd.DataFrame()

    h = history.copy()
    h["result"] = h["result"].astype(str).str.upper()
    settled = h[h["result"].isin(["WON", "LOST"])].copy()
    if settled.empty:
        return pd.DataFrame([{
            "grouping": "(nothing settled yet)",
            "value": "", "bets": 0, "won": 0, "lost": 0,
            "hit_rate": "", "avg_prob": "", "roi_per_bet": "",
        }])

    settled["won_flag"] = (settled["result"] == "WON").astype(int)
    settled["prob_num"] = pd.to_numeric(settled["prob"], errors="coerce")
    settled["profit_num"] = pd.to_numeric(settled["profit"], errors="coerce")
    settled["bucket"] = settled["odds"].map(_odds_bucket)

    rows = []

    def _summarise(grouping: str, value: str, sub: pd.DataFrame):
        n = len(sub)
        won = int(sub["won_flag"].sum())
        with_profit = sub.dropna(subset=["profit_num"])
        roi = with_profit["profit_num"].mean() if not with_profit.empty else None
        rows.append({
            "grouping": grouping,
            "value": value,
            "bets": n,
            "won": won,
            "lost": n - won,
            "hit_rate": round(won / n, 3) if n else "",
            "avg_prob": round(sub["prob_num"].mean(), 3) if sub["prob_num"].notna().any() else "",
            "roi_per_bet": round(roi, 3) if roi is not None else "",
        })

    _summarise("OVERALL", "all bets", settled)
    for bet, sub in settled.groupby("bet"):
        _summarise("by bet type", str(bet), sub)
    for src, sub in settled.groupby("source"):
        _summarise("by source tab", str(src), sub)
    for bucket, sub in settled.groupby("bucket"):
        _summarise("by odds band", str(bucket), sub)

    df = pd.DataFrame(rows)
    order = {"OVERALL": 0, "by bet type": 1, "by source tab": 2, "by odds band": 3}
    df["_o"] = df["grouping"].map(order).fillna(9)
    return df.sort_values(["_o", "bets"], ascending=[True, False]).drop(columns=["_o"]).reset_index(drop=True)


# ================= MAIN =================
def main():
    log("=== START ===")
    fd = FD(env("FOOTBALL_DATA_TOKEN"))

    today = now().date()
    f1 = today.isoformat()
    f2 = (today + timedelta(days=DAYS_AHEAD)).isoformat()
    h1 = (today - timedelta(days=HISTORY_DAYS)).isoformat()

    available_codes = set()
    try:
        for c in fd.competitions():
            code = c.get("code")
            if code:
                available_codes.add(code)
    except Exception as e:
        log(f"competitions discovery failed: {e}")

    comp_codes = list(dict.fromkeys(BASE_COMP_CODES + OPTIONAL_COMP_CODES))
    if available_codes:
        comp_codes = [c for c in comp_codes if c in available_codes] + \
                     [c for c in BASE_COMP_CODES if c not in available_codes]

    fixtures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    access_rows: List[Dict[str, Any]] = []

    for code in comp_codes:
        time.sleep(SLEEP_SECONDS)
        fx_ok, fx_n, fx_err = True, 0, ""
        try:
            fx = fd.matches(code, "SCHEDULED", f1, f2)
            fx_n = len(fx)
            log(f"{code} fixtures: {fx_n}")
            for m in fx:
                fixtures.append({
                    "league": code,
                    "utcDate": m.get("utcDate"),
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                })
        except requests.HTTPError as e:
            fx_ok = False
            status = e.response.status_code if e.response is not None else None
            fx_err = f"HTTP {status}"
            log(f"skip fixtures {code}: {fx_err}")
        except Exception as e:
            fx_ok = False
            fx_err = str(e)
            log(f"skip fixtures {code}: {fx_err}")

        time.sleep(SLEEP_SECONDS)
        rs_ok, rs_n, rs_err = True, 0, ""
        try:
            rs = fd.matches(code, "FINISHED", h1, f1)
            rs_n = len(rs)
            log(f"{code} results: {rs_n}")
            for m in sorted(rs, key=lambda x: x.get("utcDate") or ""):
                sc = ((m.get("score") or {}).get("fullTime") or {})
                hg, ag = sc.get("home"), sc.get("away")
                if hg is None or ag is None:
                    continue
                results.append({
                    "league": code,
                    "utcDate": m.get("utcDate") or "",
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                    "hg": int(hg), "ag": int(ag),
                })
        except requests.HTTPError as e:
            rs_ok = False
            status = e.response.status_code if e.response is not None else None
            rs_err = f"HTTP {status}"
            log(f"skip results {code}: {rs_err}")
        except Exception as e:
            rs_ok = False
            rs_err = str(e)
            log(f"skip results {code}: {rs_err}")

        access_rows.append({
            "code": code,
            "fixtures_ok": fx_ok, "fixtures_n": fx_n, "fixtures_err": fx_err,
            "results_ok": rs_ok, "results_n": rs_n, "results_err": rs_err,
        })

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)
    access_df = pd.DataFrame(access_rows)

    write_df(TAB_FIXTURES, sort_by_date(fx_df))
    write_df(TAB_ACCESS, access_df)

    top10_specs: List[Tuple[str, str, str]] = [
        ("p_home", "HOME WIN", "Top10_Home_Win"),
        ("p_draw", "DRAW", "Top10_Draw"),
        ("p_away", "AWAY WIN", "Top10_Away_Win"),
        ("p_btts_yes", "BTTS YES", "Top10_BTTS_Yes"),
        ("p_btts_no", "BTTS NO", "Top10_BTTS_No"),
        ("p_over_0_5", "OVER 0.5", "Top10_Over_0_5"),
        ("p_over_1_5", "OVER 1.5", "Top10_Over_1_5"),
        ("p_over_2_5", "OVER 2.5", "Top10_Over_2_5"),
        ("p_under_2_5", "UNDER 2.5", "Top10_Under_2_5"),
    ]

    visible_tabs = [
        TAB_FIXTURES, TAB_TEAM_FORM, TAB_PICKS, TAB_TOP20, TAB_SAFE, TAB_BAL,
        TAB_BEST_BETS, TAB_COMBOS, TAB_HISTORY, TAB_ACCURACY,
    ] + [t[2] for t in top10_specs]

    # Settle whatever's already logged even if today has no new fixtures —
    # results for past picks still need recording.
    history = read_df(TAB_HISTORY)
    history = settle_history(history, rs_df)

    def _finish_early(reason: str):
        write_df(TAB_HISTORY, history if history is not None and not history.empty else pd.DataFrame())
        write_df(TAB_ACCURACY, build_accuracy(history))
        set_visible_tabs(visible_tabs)
        log(f"=== DONE ({reason}) ===")

    if rs_df.empty:
        for tab in (TAB_TEAM_FORM, TAB_PICKS, TAB_TOP20, TAB_SAFE, TAB_BAL, TAB_BEST_BETS, TAB_COMBOS):
            write_df(tab, pd.DataFrame())
        for _, _, tab in top10_specs:
            write_df(tab, pd.DataFrame())
        _finish_early("no results")
        return

    league_base = compute_league_baselines(rs_df)
    team_idx = compute_team_indices(rs_df, league_base)
    write_df(TAB_TEAM_FORM, team_idx.reset_index())

    if fx_df.empty:
        for tab in (TAB_PICKS, TAB_TOP20, TAB_SAFE, TAB_BAL, TAB_BEST_BETS, TAB_COMBOS):
            write_df(tab, pd.DataFrame())
        for _, _, tab in top10_specs:
            write_df(tab, pd.DataFrame())
        _finish_early("no fixtures")
        return

    probs_rows: List[Dict[str, Any]] = []
    for _, r in fx_df.iterrows():
        league, home, away = r.get("league"), r.get("home"), r.get("away")

        base_home = safe_float(league_base.loc[league, "home_gf"], 1.35) if league in league_base.index else 1.35
        base_away = safe_float(league_base.loc[league, "away_gf"], 1.10) if league in league_base.index else 1.10
        league_base_total = base_home + base_away

        def _get(team: str, col: str, default: float = 1.0) -> float:
            try:
                return float(team_idx.loc[(league, team), col])
            except Exception:
                return default

        ha = _get(home, "home_attack")
        hd = _get(home, "home_defense")
        aa = _get(away, "away_attack")
        ad = _get(away, "away_defense")
        n_home = _get(home, "n_home", 0.0)
        n_away = _get(away, "n_away", 0.0)

        lh = base_home * ha * ad
        la = base_away * aa * hd

        dh, da = h2h_goal_adjustment(rs_df, league, home, away)
        lh = max(0.2, lh + dh)
        la = max(0.2, la + da)

        conf = confidence_score(n_home, n_away)
        p = match_probs(lh, la)

        probs_rows.append({
            "utcDate": r.get("utcDate"), "league": league, "home": home, "away": away,
            "home_xg": round(lh, 3), "away_xg": round(la, 3),
            "league_base_total": round(league_base_total, 3),
            "confidence": round(conf, 3), **p,
        })

    probs_df = pd.DataFrame(probs_rows)

    # ---- Blend model probabilities with de-vigged market odds ----
    odds_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    try:
        odds_map = fetch_market_odds(fx_df)
    except Exception as e:
        log(f"odds integration skipped: {e}")

    for i, row in probs_df.iterrows():
        key = (row["league"], row["home"], row["away"])
        mkt = market_probs_for_fixture(odds_map.get(key), row["home"], row["away"])
        for col in ("p_home", "p_draw", "p_away"):
            probs_df.at[i, col] = blend(float(row[col]), mkt.get(col))
        probs_df.at[i, "p_1x"] = probs_df.at[i, "p_home"] + probs_df.at[i, "p_draw"]
        probs_df.at[i, "p_x2"] = probs_df.at[i, "p_draw"] + probs_df.at[i, "p_away"]
        probs_df.at[i, "p_12"] = probs_df.at[i, "p_home"] + probs_df.at[i, "p_away"]
        probs_df.at[i, "has_market_odds"] = bool(mkt)

    # ---- Picks tab (informational, unfiltered) ----
    def _pick_1x2(row: pd.Series) -> Tuple[str, float]:
        opts = [("HOME", row["p_home"]), ("DRAW", row["p_draw"]), ("AWAY", row["p_away"])]
        best = max(opts, key=lambda x: float(x[1]))
        return best[0], float(best[1])

    picks = []
    for _, row in probs_df.iterrows():
        pick, pbest = _pick_1x2(row)
        picks.append({
            "utcDate": row["utcDate"], "league": row["league"],
            "home": row["home"], "away": row["away"],
            "home_xg": row["home_xg"], "away_xg": row["away_xg"],
            "confidence": row["confidence"],
            "pick_1x2": pick, "p_1x2": round(pbest, 3),
            "p_home": round(float(row["p_home"]), 3),
            "p_draw": round(float(row["p_draw"]), 3),
            "p_away": round(float(row["p_away"]), 3),
            "p_btts_yes": round(float(row["p_btts_yes"]), 3),
            "p_over_1_5": round(float(row["p_over_1_5"]), 3),
            "p_over_2_5": round(float(row["p_over_2_5"]), 3),
        })
    write_df(TAB_PICKS, sort_by_date(pd.DataFrame(picks)))

    # ---- Top10 tabs (unfiltered: these are "best by market" reference) ----
    for col, label, tab in top10_specs:
        write_df(tab, sort_by_date(top_n_for_market(probs_df, col, label, TOP_N)))

    # ---- Odds-filtered outputs ----
    log("applying odds filter (excluding anything shorter than "
        f"{MIN_DECIMAL_ODDS} = {decimal_to_fractional(MIN_DECIMAL_ODDS)}):")

    log(" Top20_Mix:")
    top20_df = attach_odds(build_top20_mix(probs_df, TOP_MIX), odds_map)
    write_df(TAB_TOP20, sort_by_date(top20_df))

    baselines = build_league_market_baselines(probs_df)

    log(" Safe_Picks:")
    safe_df = attach_odds(make_filtered_picks(probs_df, baselines, SAFE_RULES, 20, True), odds_map)
    write_df(TAB_SAFE, sort_by_date(safe_df))

    log(" Balanced_Picks:")
    bal_df = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 30, True), odds_map)
    write_df(TAB_BAL, sort_by_date(bal_df))

    log(" Best_Bets:")
    best_df = attach_odds(build_best_bets(probs_df), odds_map)
    write_df(TAB_BEST_BETS, sort_by_date(best_df))

    # ---- Combined bets (model-only probabilities, no bookmaker price) ----
    combos_df = build_combo_bets(probs_df, COMBO_TOP_N)
    write_df(TAB_COMBOS, sort_by_date(combos_df))

    # ---- Log today's picks into the immutable history, then score it ----
    history = append_new_picks(history, best_df, "Best_Bets")
    history = append_new_picks(history, safe_df, "Safe_Picks")
    history = append_new_picks(history, bal_df, "Balanced_Picks")
    history = append_new_picks(history, combos_df, "Combo_Bets")

    if history is not None and not history.empty:
        for c in HISTORY_COLS:
            if c not in history.columns:
                history[c] = ""
        history = history[HISTORY_COLS]

    write_df(TAB_HISTORY, history)
    write_df(TAB_ACCURACY, build_accuracy(history))

    set_visible_tabs(visible_tabs)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
