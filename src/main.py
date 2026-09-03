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
    OddsAPI,
    FD_TO_ODDS_SPORT,
    best_prices,
    devig_three_way,
    match_fixture,
)


# ================= CONFIG =================
BASE_COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",
    "CL", "EL", "EC",
    "ELC", "EL1", "EL2",
    "SPL", "SD",
]

OPTIONAL_COMP_CODES = [
    "DED",  # Netherlands Eredivisie
    "PPL",  # Portugal Primeira Liga
    "BSA",  # Brazil Serie A
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
MARKET_BLEND_ALPHA = 0.5  # weight on model; (1 - alpha) weight on market. Tune via backtest.

# Tabs
TAB_FIXTURES = "Fixtures"
TAB_TEAM_FORM = "Team_Form"
TAB_PICKS = "Picks"
TAB_TOP20 = "Top20_Mix"
TAB_ACCESS = "Competitions_Access"
TAB_SAFE = "Safe_Picks"
TAB_BAL = "Balanced_Picks"
TAB_BEST_BETS = "Best_Bets"


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
    if "BTTS YES & OVER" in b or "BTTS NO & UNDER" in b:
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
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_1x": p_1x,
        "p_x2": p_x2,
        "p_12": p_12,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
        "p_over_0_5": p_over_0_5,
        "p_over_1_5": p_over_1_5,
        "p_over_2_5": p_over_2_5,
        "p_over_3_5": p_over_3_5,
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
        ws.update([list(df.columns)] + df.fillna("").values.tolist())


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
    """Fetch Odds API events per league and fuzzy-match each fixture to one.
    Returns {(league, home, away): {'h2h': {...}, 'event': {...}}}."""
    api = OddsAPI(env("ODDS_API_KEY"))
    events_by_sport: Dict[str, List[Dict[str, Any]]] = {}
    matched: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for league, sub in fx_df.groupby("league"):
        sport_key = FD_TO_ODDS_SPORT.get(league)
        if not sport_key:
            continue
        if sport_key not in events_by_sport:
            try:
                events_by_sport[sport_key] = api.get_odds(sport_key)
            except Exception as e:
                log(f"odds fetch failed for {sport_key}: {e}")
                events_by_sport[sport_key] = []

        for _, r in sub.iterrows():
            ev = match_fixture(r["home"], r["away"], r["utcDate"], events_by_sport[sport_key])
            if ev is None:
                continue
            matched[(league, r["home"], r["away"])] = {
                "h2h": best_prices(ev, "h2h"),
                "event": ev,
            }

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
    "min_conf": 0.60,
    "min_prob": 0.58,
    "no_bet_low": 0.45,
    "no_bet_high": 0.55,
    "edge_win": 0.05,
    "edge_goals": 0.08,
    "max_total_xg_over_base": 1.20,
}

BAL_RULES = {
    "min_conf": 0.50,
    "min_prob": 0.55,
    "no_bet_low": 0.44,
    "no_bet_high": 0.56,
    "edge_win": 0.04,
    "edge_goals": 0.06,
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
        league = r["league"]
        home = r["home"]
        away = r["away"]
        conf = float(r["confidence"])
        home_xg = float(r["home_xg"])
        away_xg = float(r["away_xg"])
        base_tot = float(r["league_base_total"])
        tot_xg = home_xg + away_xg

        # Variance / chaos filter
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

            league_avg = base_map.get((league, mkt), None)
            if league_avg is None:
                continue
            edge = prob - league_avg

            # Different edge thresholds
            if mkt in ("HOME WIN", "DRAW", "AWAY WIN"):
                if edge < float(rules["edge_win"]):
                    continue
            else:
                if edge < float(rules["edge_goals"]):
                    continue

            sc, pen, adj = smart_score_v2(prob, conf, mkt, home_xg, away_xg, base_tot)

            picks.append({
                "utcDate": r["utcDate"],
                "league": league,
                "home": home,
                "away": away,
                "bet": mkt,
                "prob": round(prob, 3),
                "league_avg": round(league_avg, 3),
                "edge": round(edge, 3),
                "home_xg": round(home_xg, 2),
                "away_xg": round(away_xg, 2),
                "confidence": round(conf, 3),
                "adj": round(adj, 3),
                "penalty": round(pen, 3),
                "score": round(sc, 4),
            })

    df = pd.DataFrame(picks)
    if df.empty:
        return df

    df = df.sort_values(["score", "edge", "prob"], ascending=[False, False, False]).reset_index(drop=True)

    if unique_teams:
        used = set()
        keep = []
        for _, row in df.iterrows():
            h = row["home"]
            a = row["away"]
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
        h = r.get("home")
        a = r.get("away")
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
    df = probs_df[base_cols].copy()
    df = df.rename(columns={prob_col: "prob"})
    df["bet"] = bet_label

    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["prob"])

    scores = []
    for _, r in df.iterrows():
        sc, pen, adj = smart_score_v2(
            prob=float(r["prob"]),
            confidence=float(r["confidence"]),
            bet=bet_label,
            home_xg=float(r["home_xg"]),
            away_xg=float(r["away_xg"]),
            league_base_total=float(r["league_base_total"]),
        )
        scores.append((sc, pen, adj))

    df["score"] = [s[0] for s in scores]
    df["penalty"] = [s[1] for s in scores]
    df["adj"] = [s[2] for s in scores]

    df = df.sort_values(["score", "prob", "utcDate"], ascending=[False, False, True]).reset_index(drop=True)

    if UNIQUE_TEAMS_PER_TOP10:
        df = _dedupe_teams(df, top_n)
    else:
        df = df.head(top_n).reset_index(drop=True)

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

        scores = []
        for _, r in d.iterrows():
            sc, pen, adj = smart_score_v2(float(r["prob"]), float(r["confidence"]), label,
                                          float(r["home_xg"]), float(r["away_xg"]), float(r["league_base_total"]))
            scores.append((sc, pen, adj))

        d["score"] = [s[0] for s in scores]
        d["penalty"] = [s[1] for s in scores]
        d["adj"] = [s[2] for s in scores]
        return d[["utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"]]

    # NOTE: Over 0.5 excluded from mix
    parts = [
        _all_for("p_home", "HOME WIN"),
        _all_for("p_draw", "DRAW"),
        _all_for("p_away", "AWAY WIN"),
        _all_for("p_btts_yes", "BTTS YES"),
        _all_for("p_btts_no", "BTTS NO"),
        _all_for("p_over_1_5", "OVER 1.5"),
        _all_for("p_over_2_5", "OVER 2.5"),
        _all_for("p_over_3_5", "OVER 3.5"),
        _all_for("p_under_2_5", "UNDER 2.5"),
        _all_for("p_1x", "DOUBLE CHANCE 1X"),
        _all_for("p_x2", "DOUBLE CHANCE X2"),
        _all_for("p_12", "DOUBLE CHANCE 12"),
    ]

    mix = pd.concat(parts, ignore_index=True)
    mix = mix.drop_duplicates(subset=["utcDate", "home", "away", "bet"])
    mix = mix.sort_values(["score", "prob", "utcDate"], ascending=[False, False, True]).reset_index(drop=True)

    if UNIQUE_TEAMS_IN_TOP20:
        mix = _dedupe_teams(mix, top_k)
    else:
        mix = mix.head(top_k).reset_index(drop=True)

    mix.insert(0, "rank", range(1, len(mix) + 1))
    return mix[["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"]]


# ================= BEST BETS (highest win probability, blended) =================
BEST_BET_MARKETS = [
    ("p_home", "HOME WIN"),
    ("p_draw", "DRAW"),
    ("p_away", "AWAY WIN"),
    ("p_btts_yes", "BTTS YES"),
    ("p_btts_no", "BTTS NO"),
    ("p_over_1_5", "OVER 1.5"),
    ("p_over_2_5", "OVER 2.5"),
    ("p_under_2_5", "UNDER 2.5"),
    ("p_1x", "DOUBLE CHANCE 1X"),
    ("p_x2", "DOUBLE CHANCE X2"),
]


def build_best_bets(probs_df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: its single highest-probability outcome across
    all markets, then every fixture ranked together by that probability,
    descending. Confidence is used only as a tiebreaker, never to override
    the probability ranking, so the tab's meaning stays literal."""
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
            "utcDate": r["utcDate"],
            "league": r["league"],
            "home": r["home"],
            "away": r["away"],
            "bet": best_mkt,
            "probability": round(best_p, 3),
            "confidence": r["confidence"],
            "has_market_odds": bool(r.get("has_market_odds", False)),
        })

    df = pd.DataFrame(rows).sort_values(
        ["probability", "confidence"], ascending=[False, False]
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


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
        comp_codes = [c for c in comp_codes if c in available_codes] + [c for c in BASE_COMP_CODES if c not in available_codes]

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
                hg = sc.get("home")
                ag = sc.get("away")
                if hg is None or ag is None:
                    continue
                results.append({
                    "league": code,
                    "utcDate": m.get("utcDate") or "",
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                    "hg": int(hg),
                    "ag": int(ag),
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

    write_df(TAB_FIXTURES, fx_df)
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
        TAB_FIXTURES, TAB_TEAM_FORM, TAB_PICKS, TAB_TOP20, TAB_SAFE, TAB_BAL, TAB_BEST_BETS
    ] + [t[2] for t in top10_specs]

    if rs_df.empty:
        write_df(TAB_TEAM_FORM, pd.DataFrame())
        write_df(TAB_PICKS, pd.DataFrame())
        write_df(TAB_TOP20, pd.DataFrame())
        write_df(TAB_SAFE, pd.DataFrame())
        write_df(TAB_BAL, pd.DataFrame())
        write_df(TAB_BEST_BETS, pd.DataFrame())
        for _, _, tab in top10_specs:
            write_df(tab, pd.DataFrame())
        set_visible_tabs(visible_tabs)
        log("=== DONE (no results) ===")
        return

    league_base = compute_league_baselines(rs_df)
    team_idx = compute_team_indices(rs_df, league_base)
    write_df(TAB_TEAM_FORM, team_idx.reset_index())

    if fx_df.empty:
        write_df(TAB_PICKS, pd.DataFrame())
        write_df(TAB_TOP20, pd.DataFrame())
        write_df(TAB_SAFE, pd.DataFrame())
        write_df(TAB_BAL, pd.DataFrame())
        write_df(TAB_BEST_BETS, pd.DataFrame())
        for _, _, tab in top10_specs:
            write_df(tab, pd.DataFrame())
        set_visible_tabs(visible_tabs)
        log("=== DONE (no fixtures) ===")
        return

    probs_rows: List[Dict[str, Any]] = []
    for _, r in fx_df.iterrows():
        league = r.get("league")
        home = r.get("home")
        away = r.get("away")

        base_home = safe_float(league_base.loc[league, "home_gf"], 1.35) if league in league_base.index else 1.35
        base_away = safe_float(league_base.loc[league, "away_gf"], 1.10) if league in league_base.index else 1.10
        league_base_total = base_home + base_away

        def _get(team: str, col: str) -> float:
            try:
                return float(team_idx.loc[(league, team), col])
            except Exception:
                return 1.0

        def _get_n(team: str, col: str) -> float:
            try:
                return float(team_idx.loc[(league, team), col])
            except Exception:
                return 0.0

        ha = _get(home, "home_attack")
        hd = _get(home, "home_defense")
        aa = _get(away, "away_attack")
        ad = _get(away, "away_defense")

        n_home = _get_n(home, "n_home")
        n_away = _get_n(away, "n_away")

        lh = base_home * ha * ad
        la = base_away * aa * hd

        dh, da = h2h_goal_adjustment(rs_df, league, home, away)
        lh = max(0.2, lh + dh)
        la = max(0.2, la + da)

        conf = confidence_score(n_home, n_away)
        p = match_probs(lh, la)

        probs_rows.append({
            "utcDate": r.get("utcDate"),
            "league": league,
            "home": home,
            "away": away,
            "home_xg": round(lh, 3),
            "away_xg": round(la, 3),
            "league_base_total": round(league_base_total, 3),
            "confidence": round(conf, 3),
            **p,
        })

    probs_df = pd.DataFrame(probs_rows)

    # ---- Blend model probabilities with de-vigged market odds ----
    odds_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    try:
        odds_map = fetch_market_odds(fx_df)
        log(f"matched odds for {len(odds_map)}/{len(fx_df)} fixtures")
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

    # Picks tab
    def _pick_1x2(row: pd.Series) -> Tuple[str, float]:
        opts = [("HOME", row["p_home"]), ("DRAW", row["p_draw"]), ("AWAY", row["p_away"])]
        best = max(opts, key=lambda x: float(x[1]))
        return best[0], float(best[1])

    picks = []
    for _, row in probs_df.iterrows():
        pick, pbest = _pick_1x2(row)
        picks.append({
            "utcDate": row["utcDate"],
            "league": row["league"],
            "home": row["home"],
            "away": row["away"],
            "home_xg": row["home_xg"],
            "away_xg": row["away_xg"],
            "confidence": row["confidence"],
            "pick_1x2": pick,
            "p_1x2": round(pbest, 3),
            "p_home": round(float(row["p_home"]), 3),
            "p_draw": round(float(row["p_draw"]), 3),
            "p_away": round(float(row["p_away"]), 3),
            "p_btts_yes": round(float(row["p_btts_yes"]), 3),
            "p_over_1_5": round(float(row["p_over_1_5"]), 3),
            "p_over_2_5": round(float(row["p_over_2_5"]), 3),
        })
    write_df(TAB_PICKS, pd.DataFrame(picks).sort_values(["p_1x2", "utcDate"], ascending=[False, True]))

    # Top10 tabs
    for col, label, tab in top10_specs:
        write_df(tab, top_n_for_market(probs_df, col, label, TOP_N))

    # Top20 mix
    write_df(TAB_TOP20, build_top20_mix(probs_df, TOP_MIX))

    # Baselines + Safe/Balanced tabs
    baselines = build_league_market_baselines(probs_df)
    safe_df = make_filtered_picks(probs_df, baselines, SAFE_RULES, max_rows=20, unique_teams=True)
    bal_df = make_filtered_picks(probs_df, baselines, BAL_RULES, max_rows=30, unique_teams=True)

    write_df(TAB_SAFE, safe_df)
    write_df(TAB_BAL, bal_df)

    # Best Bets tab: highest win probability across all games, blended with market odds
    write_df(TAB_BEST_BETS, build_best_bets(probs_df))

    set_visible_tabs(visible_tabs)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
