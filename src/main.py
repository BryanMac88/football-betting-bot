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
    OddsApi, FD_TO_ODDS_SPORT, best_h2h_prices, named_h2h_prices,
    named_totals_prices, all_totals_prices, devig_three_way,
    match_fixture, PREFERRED_BOOKMAKERS,
)
from bet_tracking import combo_probs, COMBO_SPECS, evaluate_bet, decimal_to_fractional
from performance import apply_performance_filter, apply_league_weighting


# ================= CONFIG =================
BASE_COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1", "CL", "EL", "EC",
    "ELC", "EL1", "EL2", "NL", "SPL", "SD", "SF", "DED", "PPL",
]
OPTIONAL_COMP_CODES = ["BSA", "MLS"]

SLEEP_SECONDS = 7.0
DAYS_AHEAD = 3
HISTORY_DAYS = 210
MAX_GOALS = 10
TOP_N = 10
TOP_MIX = 20
UNIQUE_TEAMS_PER_TOP10 = True
UNIQUE_TEAMS_IN_TOP20 = True

RECENT_N = 10
RECENT_WEIGHT = 1.8
SHRINK_K = 8.0
USE_H2H = True
H2H_MATCHES_LOOKBACK = 6
H2H_SHRINK_K = 4.0
H2H_MAX_GOAL_ADJ = 0.15
H2H_BLEND = 0.35
CONF_K = 12.0
MARKET_BLEND_ALPHA = 0.5

MIN_DECIMAL_ODDS = 1.40
MAX_DECIMAL_ODDS = 4.50
ODDS_FILTER_REQUIRE_NAMED_BOOK = False
MIN_EDGE = 0.04
PERF_MIN_WIN_RATE = 0.52
PERF_MIN_BETS = 8
OVER_15_MIN_TOTAL_XG = 2.55

DEFAULT_BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.04
ARCHIVE_PENDING_DAYS = 14
SHORTLIST_SIZE = 8
SAMPLE_SIZE_WARNING = 15
FALLBACK_SIZE = 3

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
TAB_SHORTLIST = "Todays_Shortlist"
TAB_DASHBOARD = "Dashboard"
TAB_CONFIG = "Config"


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
        return default if math.isnan(v) else v
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def kelly_stake(prob: float, odds: float, bankroll: float,
                fraction: float = KELLY_FRACTION, max_pct: float = MAX_STAKE_PCT) -> float:
    try:
        p, o = float(prob), float(odds)
    except (TypeError, ValueError):
        return 0.0
    if o <= 1.0 or p <= 0 or p >= 1:
        return 0.0
    b = o - 1.0
    full = (p * (b + 1) - 1) / b
    if full <= 0:
        return 0.0
    stake = bankroll * fraction * full
    return round(min(stake, bankroll * max_pct), 2)


# ================= SMART SCORE =================
def market_weight(bet: str) -> float:
    b = bet.upper()
    if b in ("HOME WIN", "AWAY WIN"): return 1.00
    if b == "OVER 1.5": return 1.08
    if b == "DRAW": return 0.88
    if b.startswith("DOUBLE CHANCE"): return 0.92
    if b in ("BTTS YES", "BTTS NO"): return 0.94
    if b.startswith("OVER ") or b.startswith("UNDER "): return 0.92
    if "&" in b: return 0.88
    return 0.90

def smart_score_v2(prob, confidence, bet, home_xg, away_xg, league_base_total):
    p = clamp(float(prob), 0.0, 1.0)
    c = clamp(float(confidence), 0.0, 1.0)
    tot = max(0.1, float(home_xg) + float(away_xg))
    base_tot = max(0.1, float(league_base_total))
    gd = float(home_xg) - float(away_xg)
    extremeness = abs(p - 0.5) * 2.0
    penalty = clamp((extremeness ** 1.25) * (1.0 - c) * 0.28, 0.0, 0.22)
    adj = 1.0
    b = bet.upper()
    if b == "DRAW":
        adj *= (1.0 - clamp((tot / base_tot - 1.0) * 0.18, 0.0, 0.18))
        adj *= (1.0 - clamp(abs(gd) * 0.10, 0.0, 0.20))
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
        tot_adj = 1.0 + clamp((tot / base_tot - 1.0) * 0.06, -0.06, 0.06) if b == "BTTS YES" else \
                  1.0 + clamp((1.0 - tot / base_tot) * 0.06, -0.06, 0.06)
        adj *= bal * tot_adj
    adj = clamp(adj, 0.80, 1.18)
    score = (p * c) * market_weight(bet) * adj * (1.0 - penalty)
    return float(score), float(penalty), float(adj)


# ================= GOAL MODEL =================
def match_probs(lh: float, la: float) -> Dict[str, float]:
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]
    p_home = p_draw = p_away = p_btts_yes = 0.0
    p_over_0_5 = p_over_1_5 = p_over_2_5 = p_over_3_5 = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i > 0 and j > 0: p_btts_yes += p
            tg = i + j
            if tg > 0: p_over_0_5 += p
            if tg > 1: p_over_1_5 += p
            if tg > 2: p_over_2_5 += p
            if tg > 3: p_over_3_5 += p
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_1x": p_home + p_draw, "p_x2": p_draw + p_away, "p_12": p_home + p_away,
        "p_btts_yes": p_btts_yes, "p_btts_no": 1.0 - p_btts_yes,
        "p_over_0_5": p_over_0_5, "p_over_1_5": p_over_1_5,
        "p_over_2_5": p_over_2_5, "p_over_3_5": p_over_3_5,
        "p_under_1_5": 1.0 - p_over_1_5, "p_under_2_5": 1.0 - p_over_2_5,
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

def _get_or_create_ws(sh, title, rows=4000, cols=60):
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
        ws.update([list(df.columns)] + df.fillna("").astype(str).values.tolist(),
                  value_input_option="USER_ENTERED")

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

def sort_by_date(df, date_col="utcDate"):
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
        title, sid = props.get("title"), props.get("sheetId")
        if title is None or sid is None:
            continue
        reqs.append({
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "hidden": title not in keep_titles},
                "fields": "hidden",
            }
        })
    if reqs:
        sh.batch_update({"requests": reqs})

def get_bankroll() -> float:
    try:
        sh = open_sheet()
        ws = sh.worksheet(TAB_CONFIG)
        val = ws.acell("B2").value
        return float(val) if val else DEFAULT_BANKROLL
    except Exception:
        return DEFAULT_BANKROLL


# ================= FOOTBALL-DATA =================
@dataclass
class FD:
    token: str
    base: str = "https://api.football-data.org/v4"
    def get(self, path, params=None):
        r = requests.get(self.base + path, headers={"X-Auth-Token": self.token},
                         params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()
    def matches(self, code, status, d1, d2):
        return self.get(f"/competitions/{code}/matches",
                        {"status": status, "dateFrom": d1, "dateTo": d2}).get("matches", [])
    def competitions(self):
        return self.get("/competitions", {}).get("competitions", [])


# ================= MODEL CORE =================
def recency_weighted_mean(values, recent_n=RECENT_N, recent_weight=RECENT_WEIGHT):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    n = len(vals)
    weights = [1.0] * n
    for i in range(max(0, n - recent_n), n):
        weights[i] = recent_weight
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

def shrink(mean_est, n, prior_mean, k=SHRINK_K):
    if n <= 0 or math.isnan(mean_est):
        return prior_mean
    return (n * mean_est + k * prior_mean) / (n + k)

def compute_league_baselines(rs):
    g = rs.groupby("league")
    return pd.DataFrame({"home_gf": g["hg"].mean(), "away_gf": g["ag"].mean(), "n": g.size()})

def compute_team_indices(rs, league_baselines):
    rs = rs.sort_values("utcDate")
    rows = []
    for league, sub in rs.groupby("league"):
        base_home = safe_float(league_baselines.loc[league, "home_gf"], 1.35) if league in league_baselines.index else 1.35
        base_away = safe_float(league_baselines.loc[league, "away_gf"], 1.10) if league in league_baselines.index else 1.10
        for team, tsub in sub.groupby("home"):
            hg, ag = tsub["hg"].tolist(), tsub["ag"].tolist()
            n = len(hg)
            rows.append({"league": league, "team": team,
                         "home_hg": shrink(recency_weighted_mean(hg), n, base_home),
                         "home_ag": shrink(recency_weighted_mean(ag), n, base_away), "n_home": n})
        for team, tsub in sub.groupby("away"):
            ag, hg = tsub["ag"].tolist(), tsub["hg"].tolist()
            n = len(ag)
            rows.append({"league": league, "team": team,
                         "away_hg": shrink(recency_weighted_mean(ag), n, base_away),
                         "away_ag": shrink(recency_weighted_mean(hg), n, base_home), "n_away": n})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(["league", "team"], as_index=False).agg({
        "home_hg": "max", "home_ag": "max", "away_hg": "max", "away_ag": "max",
        "n_home": "max", "n_away": "max"}).fillna(0.0)
    out = []
    for _, r in agg.iterrows():
        league = r["league"]
        base_home = safe_float(league_baselines.loc[league, "home_gf"], 1.35) if league in league_baselines.index else 1.35
        base_away = safe_float(league_baselines.loc[league, "away_gf"], 1.10) if league in league_baselines.index else 1.10
        out.append({
            "league": league, "team": r["team"],
            "home_attack": clamp(r["home_hg"] / base_home if base_home else 1.0, 0.55, 1.75),
            "home_defense": clamp(r["home_ag"] / base_away if base_away else 1.0, 0.55, 1.75),
            "away_attack": clamp(r["away_hg"] / base_away if base_away else 1.0, 0.55, 1.75),
            "away_defense": clamp(r["away_ag"] / base_home if base_home else 1.0, 0.55, 1.75),
            "n_home": float(r["n_home"]), "n_away": float(r["n_away"]),
        })
    return pd.DataFrame(out).set_index(["league", "team"])

def h2h_goal_adjustment(rs, league, home, away):
    if not USE_H2H or rs.empty:
        return 0.0, 0.0
    sub = rs[(rs["league"] == league) & (
        ((rs["home"] == home) & (rs["away"] == away)) |
        ((rs["home"] == away) & (rs["away"] == home)))].sort_values("utcDate").tail(H2H_MATCHES_LOOKBACK)
    if sub.empty:
        return 0.0, 0.0
    hg, ag = [], []
    for _, r in sub.iterrows():
        if r["home"] == home:
            hg.append(float(r["hg"])); ag.append(float(r["ag"]))
        else:
            hg.append(float(r["ag"])); ag.append(float(r["hg"]))
    n = len(hg)
    if n == 0:
        return 0.0, 0.0
    hmean = recency_weighted_mean(hg, min(RECENT_N, n), RECENT_WEIGHT)
    amean = recency_weighted_mean(ag, min(RECENT_N, n), RECENT_WEIGHT)
    mean_total = (hmean + amean) / 2.0
    factor = (n / (n + H2H_SHRINK_K)) * H2H_BLEND
    return (clamp((hmean - mean_total) * factor, -H2H_MAX_GOAL_ADJ, H2H_MAX_GOAL_ADJ),
            clamp((amean - mean_total) * factor, -H2H_MAX_GOAL_ADJ, H2H_MAX_GOAL_ADJ))

def confidence_score(n_home, n_away):
    n = max(0.0, float(n_home or 0) + float(n_away or 0))
    return float(n / (n + CONF_K))


# ================= ODDS =================
def fetch_market_odds(fx_df):
    api = OddsApi(env("ODDS_API_KEY"))
    events_by_sport = {}
    matched = {}
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
                "h2h": best_h2h_prices(ev), "named_h2h": named_h2h,
                "named_totals": named_tot, "all_totals": all_totals_prices(ev),
                "book": book_h2h or book_tot, "event": ev,
            }
    log(f"odds matched: {len(matched)} fixtures")
    return matched

def market_probs_for_fixture(odds_entry, home, away):
    out = {}
    if not odds_entry:
        return out
    h2h = odds_entry.get("h2h", {})
    if home in h2h and away in h2h and "Draw" in h2h:
        ph, pdw, pa = devig_three_way(h2h[home], h2h["Draw"], h2h[away])
        out["p_home"], out["p_draw"], out["p_away"] = ph, pdw, pa
    return out

def blend(model_p, market_p, alpha=MARKET_BLEND_ALPHA):
    return model_p if market_p is None else alpha * model_p + (1 - alpha) * market_p

def odds_for_bet(odds_entry, bet, home, away):
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
        if key in named_h2h: return named_h2h[key], book or "named"
        if key in all_h2h: return all_h2h[key], "market-best"
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

def passes_odds_filter(price, source):
    if price is None:
        return not ODDS_FILTER_REQUIRE_NAMED_BOOK
    p = float(price)
    return MIN_DECIMAL_ODDS <= p <= MAX_DECIMAL_ODDS

def attach_odds(df, odds_map, bankroll, apply_filter=True):
    if df is None or df.empty:
        return df
    prices, fracs, sources, keeps, stakes = [], [], [], [], []
    for _, r in df.iterrows():
        entry = odds_map.get((r.get("league"), r.get("home"), r.get("away")))
        price, source = odds_for_bet(entry, str(r.get("bet", "")), r.get("home"), r.get("away"))
        prices.append(round(price, 3) if price else "")
        fracs.append(decimal_to_fractional(price) if price else "")
        sources.append(source)
        keeps.append(passes_odds_filter(price, source))
        stakes.append(kelly_stake(r.get("prob"), price, bankroll) if price else "")
    out = df.copy()
    out["odds"] = prices
    out["fractional"] = fracs
    out["odds_source"] = sources
    out["stake"] = stakes
    out["closing_odds"] = ""
    if apply_filter:
        before = len(out)
        out = out[pd.Series(keeps, index=out.index)].reset_index(drop=True)
        if before - len(out):
            log(f"  odds filter removed {before - len(out)} pick(s)")
        if "rank" in out.columns and not out.empty:
            out["rank"] = range(1, len(out) + 1)
    return out

def apply_value_filter(df, min_edge=MIN_EDGE):
    if df is None or df.empty or "odds" not in df.columns or "prob" not in df.columns:
        return df
    out = df.copy()
    out["_odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out["_prob"] = pd.to_numeric(out["prob"], errors="coerce")
    out["_edge"] = out["_prob"] - (1.0 / out["_odds"])
    before = len(out)
    out = out[out["_edge"] >= min_edge].copy()
    if before - len(out):
        log(f"  value filter removed {before - len(out)} pick(s)")
    out = out.drop(columns=["_odds", "_prob", "_edge"], errors="ignore")
    if "rank" in out.columns and not out.empty:
        out = out.sort_values("score" if "score" in out.columns else "rank", ascending=False)
        out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)

def apply_over15_quality_filter(df):
    if df is None or df.empty or "bet" not in df.columns:
        return df
    out = df.copy()
    mask = out["bet"].astype(str).str.upper() != "OVER 1.5"
    if "home_xg" in out.columns and "away_xg" in out.columns:
        total = pd.to_numeric(out["home_xg"], errors="coerce") + pd.to_numeric(out["away_xg"], errors="coerce")
        mask = mask | (total >= OVER_15_MIN_TOTAL_XG)
    before = len(out)
    out = out[mask].copy()
    if before - len(out):
        log(f"  Over 1.5 quality filter removed {before - len(out)} picks")
    if "rank" in out.columns and not out.empty:
        out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


# ================= NEVER-EMPTY FALLBACK =================
def ensure_at_least_one(strict_df: pd.DataFrame, soft_df: pd.DataFrame, size: int = FALLBACK_SIZE) -> pd.DataFrame:
    if strict_df is not None and not strict_df.empty:
        out = strict_df.copy()
        if "warning" not in out.columns:
            out["warning"] = ""
        return out

    if soft_df is None or soft_df.empty:
        return pd.DataFrame()

    out = soft_df.copy()
    if "score" in out.columns:
        out = out.sort_values("score", ascending=False)
    elif "prob" in out.columns:
        out = out.sort_values("prob", ascending=False)

    out = out.head(size).reset_index(drop=True)
    out["warning"] = "BELOW ACCEPTABLE LEVEL – bet at your own risk"
    out["rank"] = range(1, len(out) + 1)
    log(f"  fallback used – showing {len(out)} lower-quality pick(s)")
    return out


# ================= PICK BUILDERS =================
MARKET_COLS = {
    "HOME WIN": "p_home", "DRAW": "p_draw", "AWAY WIN": "p_away",
    "BTTS YES": "p_btts_yes", "BTTS NO": "p_btts_no",
    "OVER 1.5": "p_over_1_5", "OVER 2.5": "p_over_2_5",
}
SAFE_RULES = {"min_conf": 0.60, "min_prob": 0.58, "no_bet_low": 0.45, "no_bet_high": 0.55,
              "edge_win": 0.05, "edge_goals": 0.08, "max_total_xg_over_base": 1.20}
BAL_RULES = {"min_conf": 0.50, "min_prob": 0.55, "no_bet_low": 0.44, "no_bet_high": 0.56,
             "edge_win": 0.04, "edge_goals": 0.06, "max_total_xg_over_base": 1.50}

def build_league_market_baselines(probs_df):
    rows = []
    for league, sub in probs_df.groupby("league"):
        for mkt, col in MARKET_COLS.items():
            if col in sub.columns:
                rows.append({"league": league, "market": mkt, "league_avg_prob": float(sub[col].mean())})
    return pd.DataFrame(rows)

def make_filtered_picks(probs_df, baselines, rules, max_rows=30, unique_teams=True):
    if probs_df.empty or baselines.empty:
        return pd.DataFrame()
    base_map = {(r["league"], r["market"]): float(r["league_avg_prob"]) for _, r in baselines.iterrows()}
    picks = []
    for _, r in probs_df.iterrows():
        league, home, away = r["league"], r["home"], r["away"]
        conf, home_xg, away_xg = float(r["confidence"]), float(r["home_xg"]), float(r["away_xg"])
        base_tot, tot_xg = float(r["league_base_total"]), home_xg + away_xg
        if tot_xg > base_tot + float(rules["max_total_xg_over_base"]):
            continue
        for mkt, col in MARKET_COLS.items():
            prob = float(r[col])
            if prob < float(rules["min_prob"]) or float(rules["no_bet_low"]) < prob < float(rules["no_bet_high"]):
                continue
            if conf < float(rules["min_conf"]):
                continue
            league_avg = base_map.get((league, mkt))
            if league_avg is None:
                continue
            edge = prob - league_avg
            min_edge = float(rules["edge_goals"]) * 0.75 if mkt == "OVER 1.5" else \
                       float(rules["edge_win"]) if mkt in ("HOME WIN", "DRAW", "AWAY WIN") else float(rules["edge_goals"])
            if edge < min_edge:
                continue
            sc, pen, adj = smart_score_v2(prob, conf, mkt, home_xg, away_xg, base_tot)
            picks.append({"utcDate": r["utcDate"], "league": league, "home": home, "away": away,
                          "bet": mkt, "prob": round(prob, 3), "league_avg": round(league_avg, 3),
                          "edge": round(edge, 3), "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2),
                          "confidence": round(conf, 3), "adj": round(adj, 3), "penalty": round(pen, 3),
                          "score": round(sc, 4)})
    df = pd.DataFrame(picks)
    if df.empty:
        return df
    df = df.sort_values(["score", "edge", "prob"], ascending=[False, False, False]).reset_index(drop=True)
    if unique_teams:
        used, keep = set(), []
        for _, row in df.iterrows():
            if row["home"] in used or row["away"] in used:
                continue
            keep.append(row)
            used.add(row["home"]); used.add(row["away"])
            if len(keep) >= max_rows:
                break
        df = pd.DataFrame(keep)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

CORE_COLS = ["utcDate", "league", "home", "away"]

def _dedupe_teams(df, limit):
    used, kept = set(), []
    for _, r in df.iterrows():
        h, a = r.get("home"), r.get("away")
        if h in used or a in used:
            continue
        kept.append(r)
        if h: used.add(h)
        if a: used.add(a)
        if len(kept) >= limit:
            break
    return pd.DataFrame(kept).reset_index(drop=True) if kept else df.head(0)

def top_n_for_market(probs_df, prob_col, bet_label, top_n=TOP_N):
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()
    df = probs_df[CORE_COLS + ["confidence", "home_xg", "away_xg", "league_base_total", prob_col]].copy()
    df = df.rename(columns={prob_col: "prob"})
    df["bet"] = bet_label
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["prob"])
    scores = [smart_score_v2(float(r["prob"]), float(r["confidence"]), bet_label,
                             float(r["home_xg"]), float(r["away_xg"]), float(r["league_base_total"]))
              for _, r in df.iterrows()]
    df["score"] = [s[0] for s in scores]
    df["penalty"] = [s[1] for s in scores]
    df["adj"] = [s[2] for s in scores]
    df = df.sort_values(["score", "prob"], ascending=[False, False]).reset_index(drop=True)
    df = _dedupe_teams(df, top_n) if UNIQUE_TEAMS_PER_TOP10 else df.head(top_n)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"]]

def build_best_bets(probs_df):
    if probs_df.empty:
        return pd.DataFrame()
    markets = [("p_home", "HOME WIN"), ("p_draw", "DRAW"), ("p_away", "AWAY WIN"),
               ("p_btts_yes", "BTTS YES"), ("p_btts_no", "BTTS NO"),
               ("p_over_1_5", "OVER 1.5"), ("p_over_2_5", "OVER 2.5"),
               ("p_1x", "DOUBLE CHANCE 1X"), ("p_x2", "DOUBLE CHANCE X2")]
    rows = []
    for _, r in probs_df.iterrows():
        best_mkt, best_p = None, -1.0
        for col, label in markets:
            p = float(r.get(col, 0))
            if p > best_p:
                best_p, best_mkt = p, label
        rows.append({"utcDate": r["utcDate"], "league": r["league"], "home": r["home"], "away": r["away"],
                     "bet": best_mkt, "prob": round(best_p, 3), "confidence": r["confidence"],
                     "home_xg": r.get("home_xg"), "away_xg": r.get("away_xg")})
    df = pd.DataFrame(rows).sort_values(["prob", "confidence"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

def build_combo_bets(probs_df, top_k=COMBO_TOP_N):
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in probs_df.iterrows():
        lh, la, conf, base_tot = float(r["home_xg"]), float(r["away_xg"]), float(r["confidence"]), float(r["league_base_total"])
        for label, prob in combo_probs(lh, la, MAX_GOALS).items():
            if prob < COMBO_MIN_PROB:
                continue
            sc, _, _ = smart_score_v2(prob, conf, label, lh, la, base_tot)
            rows.append({"utcDate": r["utcDate"], "league": r["league"], "home": r["home"], "away": r["away"],
                         "bet": label, "prob": round(prob, 3), "model_implied_odds": round(1/prob, 2) if prob else "",
                         "confidence": round(conf, 3), "home_xg": round(lh, 2), "away_xg": round(la, 2),
                         "score": round(sc, 4)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["score", "prob"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

def build_shortlist(best_df, safe_df, bal_df, size=SHORTLIST_SIZE):
    frames = []
    for df, src in [(best_df, "Best"), (safe_df, "Safe"), (bal_df, "Balanced")]:
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp["source"] = src
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()

    mix = pd.concat(frames, ignore_index=True)
    mix = mix.drop_duplicates(subset=["utcDate", "home", "away", "bet"])

    # Drop existing rank column if present (this was the bug)
    if "rank" in mix.columns:
        mix = mix.drop(columns=["rank"])

    if "score" in mix.columns:
        mix = mix.sort_values("score", ascending=False)
    elif "prob" in mix.columns:
        mix = mix.sort_values("prob", ascending=False)

    mix = mix.head(size).reset_index(drop=True)
    mix.insert(0, "rank", range(1, len(mix) + 1))

    keep = ["rank", "utcDate", "league", "home", "away", "bet", "prob", "odds", "fractional",
            "stake", "confidence", "score", "source", "odds_source", "warning"]
    return mix[[c for c in keep if c in mix.columns]]

def build_dashboard(shortlist, history, bankroll):
    rows = []
    rows.append({"metric": "Run time (UTC)", "value": now().strftime("%Y-%m-%d %H:%M")})
    rows.append({"metric": "Bankroll", "value": f"{bankroll:.0f}"})
    rows.append({"metric": "Shortlist size", "value": len(shortlist) if shortlist is not None else 0})
    if shortlist is not None and not shortlist.empty and "stake" in shortlist.columns:
        total_stake = pd.to_numeric(shortlist["stake"], errors="coerce").sum()
        rows.append({"metric": "Total recommended stake", "value": f"{total_stake:.2f}"})
        rows.append({"metric": "% of bankroll", "value": f"{(total_stake / bankroll * 100):.1f}%"})
    if history is not None and not history.empty:
        settled = history[history["result"].astype(str).str.upper().isin(["WON", "LOST"])]
        if not settled.empty:
            wins = (settled["result"].astype(str).str.upper() == "WON").sum()
            rows.append({"metric": "Historical hit rate", "value": f"{wins / len(settled) * 100:.0f}%"})
            profit = pd.to_numeric(settled.get("profit"), errors="coerce").sum()
            rows.append({"metric": "Historical profit (units)", "value": f"{profit:.2f}"})
    return pd.DataFrame(rows)


# ================= HISTORY / ACCURACY =================
HISTORY_COLS = [
    "logged_utc", "utcDate", "league", "home", "away", "bet", "source",
    "prob", "odds", "fractional", "odds_source", "stake", "closing_odds",
    "result", "score", "profit",
]

def _hist_key(row):
    return (str(row.get("utcDate", ""))[:10], str(row.get("home", "")),
            str(row.get("away", "")), str(row.get("bet", "")), str(row.get("source", "")))

def append_new_picks(history, picks, source):
    if picks is None or picks.empty:
        return history
    existing = {_hist_key(h) for _, h in history.iterrows()} if not history.empty else set()
    new_rows = []
    stamp = now().strftime("%Y-%m-%d %H:%M:%S UTC")
    for _, p in picks.iterrows():
        cand = {"utcDate": p.get("utcDate", ""), "home": p.get("home", ""),
                "away": p.get("away", ""), "bet": p.get("bet", ""), "source": source}
        if _hist_key(cand) in existing:
            continue
        new_rows.append({
            "logged_utc": stamp, "utcDate": p.get("utcDate", ""), "league": p.get("league", ""),
            "home": p.get("home", ""), "away": p.get("away", ""), "bet": p.get("bet", ""),
            "source": source, "prob": p.get("prob", ""), "odds": p.get("odds", ""),
            "fractional": p.get("fractional", ""), "odds_source": p.get("odds_source", ""),
            "stake": p.get("stake", ""), "closing_odds": "", "result": "PENDING",
            "score": "", "profit": "",
        })
    if not new_rows:
        return history
    log(f"  logging {len(new_rows)} new pick(s) from {source}")
    add = pd.DataFrame(new_rows)
    return add if history.empty else pd.concat([history, add], ignore_index=True)

def settle_history(history, rs_df):
    if history is None or history.empty or rs_df is None or rs_df.empty:
        return history
    results = {}
    for _, r in rs_df.iterrows():
        key = (str(r["utcDate"])[:10], str(r["home"]), str(r["away"]))
        try:
            results[key] = (int(r["hg"]), int(r["ag"]))
        except Exception:
            continue
    settled = 0
    out = history.copy()
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
            odds = float(h.get("odds") or 0)
        except Exception:
            odds = 0.0
        out.at[i, "profit"] = round(odds - 1, 3) if won and odds > 1 else (-1.0 if odds > 1 else "")
        settled += 1
    if settled:
        log(f"  settled {settled} bet(s)")
    return out

def archive_old_pending(history, days=ARCHIVE_PENDING_DAYS):
    if history is None or history.empty:
        return history
    out = history.copy()
    cutoff = (now() - timedelta(days=days)).strftime("%Y-%m-%d")
    mask = (out["result"].astype(str).str.upper() == "PENDING") & \
           (out["utcDate"].astype(str).str[:10] < cutoff)
    removed = mask.sum()
    if removed:
        log(f"  archived {removed} old PENDING bets")
        out = out[~mask].reset_index(drop=True)
    return out

def build_accuracy(history):
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    h["result"] = h["result"].astype(str).str.upper()
    settled = h[h["result"].isin(["WON", "LOST"])].copy()
    if settled.empty:
        return pd.DataFrame([{"grouping": "(nothing settled yet)", "value": "", "bets": 0,
                              "won": 0, "lost": 0, "hit_rate": "", "avg_prob": "", "roi_per_bet": "", "note": ""}])
    settled["won_flag"] = (settled["result"] == "WON").astype(int)
    settled["prob_num"] = pd.to_numeric(settled["prob"], errors="coerce")
    settled["profit_num"] = pd.to_numeric(settled["profit"], errors="coerce")
    rows = []
    def _summ(grouping, value, sub):
        n = len(sub)
        won = int(sub["won_flag"].sum())
        roi = sub["profit_num"].mean() if sub["profit_num"].notna().any() else None
        note = "LOW SAMPLE" if n < SAMPLE_SIZE_WARNING else ""
        rows.append({"grouping": grouping, "value": value, "bets": n, "won": won, "lost": n - won,
                     "hit_rate": f"{round(won / n * 100)}%" if n else "",
                     "avg_prob": round(sub["prob_num"].mean(), 3) if sub["prob_num"].notna().any() else "",
                     "roi_per_bet": round(roi, 3) if roi is not None else "", "note": note})
    _summ("OVERALL", "all bets", settled)
    for bet, sub in settled.groupby("bet"):
        _summ("by bet type", str(bet), sub)
    for src, sub in settled.groupby("source"):
        _summ("by source tab", str(src), sub)
    df = pd.DataFrame(rows)
    order = {"OVERALL": 0, "by bet type": 1, "by source tab": 2}
    df["_o"] = df["grouping"].map(order).fillna(9)
    return df.sort_values(["_o", "bets"], ascending=[True, False]).drop(columns=["_o"]).reset_index(drop=True)


# ================= MAIN =================
def main():
    log("=== START ===")
    bankroll = get_bankroll()
    log(f"Using bankroll: {bankroll}")

    fd = FD(env("FOOTBALL_DATA_TOKEN"))
    today = now().date()
    f1, f2 = today.isoformat(), (today + timedelta(days=DAYS_AHEAD)).isoformat()
    h1 = (today - timedelta(days=HISTORY_DAYS)).isoformat()

    available = set()
    try:
        for c in fd.competitions():
            if c.get("code"):
                available.add(c["code"])
    except Exception as e:
        log(f"competitions discovery failed: {e}")

    codes = list(dict.fromkeys(BASE_COMP_CODES + OPTIONAL_COMP_CODES))
    if available:
        codes = [c for c in codes if c in available] + [c for c in BASE_COMP_CODES if c not in available]

    fixtures, results, access_rows = [], [], []
    for code in codes:
        time.sleep(SLEEP_SECONDS)
        try:
            fx = fd.matches(code, "SCHEDULED", f1, f2)
            log(f"{code} fixtures: {len(fx)}")
            for m in fx:
                fixtures.append({"league": code, "utcDate": m.get("utcDate"),
                                 "home": (m.get("homeTeam") or {}).get("name"),
                                 "away": (m.get("awayTeam") or {}).get("name")})
            fx_ok, fx_n, fx_err = True, len(fx), ""
        except Exception as e:
            fx_ok, fx_n, fx_err = False, 0, str(e)
            log(f"skip fixtures {code}: {e}")

        time.sleep(SLEEP_SECONDS)
        try:
            rs = fd.matches(code, "FINISHED", h1, f1)
            log(f"{code} results: {len(rs)}")
            for m in sorted(rs, key=lambda x: x.get("utcDate") or ""):
                sc = ((m.get("score") or {}).get("fullTime") or {})
                if sc.get("home") is None or sc.get("away") is None:
                    continue
                results.append({"league": code, "utcDate": m.get("utcDate") or "",
                                "home": (m.get("homeTeam") or {}).get("name"),
                                "away": (m.get("awayTeam") or {}).get("name"),
                                "hg": int(sc["home"]), "ag": int(sc["away"])})
            rs_ok, rs_n, rs_err = True, len(rs), ""
        except Exception as e:
            rs_ok, rs_n, rs_err = False, 0, str(e)
            log(f"skip results {code}: {e}")

        access_rows.append({"code": code, "fixtures_ok": fx_ok, "fixtures_n": fx_n, "fixtures_err": fx_err,
                            "results_ok": rs_ok, "results_n": rs_n, "results_err": rs_err})

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)
    write_df(TAB_FIXTURES, sort_by_date(fx_df))
    write_df(TAB_ACCESS, pd.DataFrame(access_rows))

    top10_specs = [("p_over_1_5", "OVER 1.5", "Top10_Over_1_5"),
                   ("p_over_2_5", "OVER 2.5", "Top10_Over_2_5")]
    visible_tabs = [TAB_BEST_BETS, "Top10_Over_1_5", "Top10_Over_2_5", TAB_COMBOS,
                    TAB_SAFE, TAB_BAL, TAB_SHORTLIST, TAB_DASHBOARD, TAB_ACCURACY]

    history = read_df(TAB_HISTORY)
    history = settle_history(history, rs_df)
    history = archive_old_pending(history)

    def _finish(reason):
        write_df(TAB_HISTORY, history if history is not None and not history.empty else pd.DataFrame())
        write_df(TAB_ACCURACY, build_accuracy(history))
        set_visible_tabs(visible_tabs)
        log(f"=== DONE ({reason}) ===")

    if rs_df.empty:
        for t in (TAB_TEAM_FORM, TAB_PICKS, TAB_SAFE, TAB_BAL, TAB_BEST_BETS, TAB_COMBOS, TAB_SHORTLIST, TAB_DASHBOARD):
            write_df(t, pd.DataFrame())
        for _, _, t in top10_specs:
            write_df(t, pd.DataFrame())
        _finish("no results")
        return

    league_base = compute_league_baselines(rs_df)
    team_idx = compute_team_indices(rs_df, league_base)
    write_df(TAB_TEAM_FORM, team_idx.reset_index())

    if fx_df.empty:
        for t in (TAB_PICKS, TAB_SAFE, TAB_BAL, TAB_BEST_BETS, TAB_COMBOS, TAB_SHORTLIST, TAB_DASHBOARD):
            write_df(t, pd.DataFrame())
        for _, _, t in top10_specs:
            write_df(t, pd.DataFrame())
        _finish("no fixtures")
        return

    probs_rows = []
    for _, r in fx_df.iterrows():
        league, home, away = r["league"], r["home"], r["away"]
        base_home = safe_float(league_base.loc[league, "home_gf"], 1.35) if league in league_base.index else 1.35
        base_away = safe_float(league_base.loc[league, "away_gf"], 1.10) if league in league_base.index else 1.10
        def _get(team, col, default=1.0):
            try: return float(team_idx.loc[(league, team), col])
            except: return default
        ha, hd, aa, ad = _get(home, "home_attack"), _get(home, "home_defense"), _get(away, "away_attack"), _get(away, "away_defense")
        n_home, n_away = _get(home, "n_home", 0), _get(away, "n_away", 0)
        lh = max(0.2, base_home * ha * ad)
        la = max(0.2, base_away * aa * hd)
        dh, da = h2h_goal_adjustment(rs_df, league, home, away)
        lh, la = max(0.2, lh + dh), max(0.2, la + da)
        conf = confidence_score(n_home, n_away)
        p = match_probs(lh, la)
        probs_rows.append({"utcDate": r["utcDate"], "league": league, "home": home, "away": away,
                           "home_xg": round(lh, 3), "away_xg": round(la, 3),
                           "league_base_total": round(base_home + base_away, 3),
                           "confidence": round(conf, 3), **p})
    probs_df = pd.DataFrame(probs_rows)

    odds_map = {}
    try:
        odds_map = fetch_market_odds(fx_df)
    except Exception as e:
        log(f"odds integration skipped: {e}")

    for i, row in probs_df.iterrows():
        mkt = market_probs_for_fixture(odds_map.get((row["league"], row["home"], row["away"])), row["home"], row["away"])
        for col in ("p_home", "p_draw", "p_away"):
            probs_df.at[i, col] = blend(float(row[col]), mkt.get(col))
        probs_df.at[i, "p_1x"] = probs_df.at[i, "p_home"] + probs_df.at[i, "p_draw"]
        probs_df.at[i, "p_x2"] = probs_df.at[i, "p_draw"] + probs_df.at[i, "p_away"]
        probs_df.at[i, "p_12"] = probs_df.at[i, "p_home"] + probs_df.at[i, "p_away"]

    # ---------- Top10 tabs ----------
    for col, label, tab in top10_specs:
        soft = top_n_for_market(probs_df, col, label, TOP_N)
        final = ensure_at_least_one(soft, soft, size=3)
        write_df(tab, sort_by_date(final))

    baselines = build_league_market_baselines(probs_df)

    # ---------- Safe_Picks ----------
    log("Safe_Picks...")
    soft_safe = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 30), odds_map, bankroll, apply_filter=False)
    strict_safe = attach_odds(make_filtered_picks(probs_df, baselines, SAFE_RULES, 20), odds_map, bankroll)
    strict_safe = apply_value_filter(strict_safe)
    strict_safe = apply_over15_quality_filter(strict_safe)
    strict_safe = apply_performance_filter(strict_safe, history)
    strict_safe = apply_league_weighting(strict_safe, history)
    safe_df = ensure_at_least_one(strict_safe, soft_safe)
    write_df(TAB_SAFE, sort_by_date(safe_df))

    # ---------- Balanced_Picks ----------
    log("Balanced_Picks...")
    soft_bal = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 40), odds_map, bankroll, apply_filter=False)
    strict_bal = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 30), odds_map, bankroll)
    strict_bal = apply_value_filter(strict_bal)
    strict_bal = apply_over15_quality_filter(strict_bal)
    strict_bal = apply_performance_filter(strict_bal, history)
    strict_bal = apply_league_weighting(strict_bal, history)
    bal_df = ensure_at_least_one(strict_bal, soft_bal)
    write_df(TAB_BAL, sort_by_date(bal_df))

    # ---------- Best_Bets ----------
    log("Best_Bets...")
    soft_best = attach_odds(build_best_bets(probs_df), odds_map, bankroll, apply_filter=False)
    strict_best = attach_odds(build_best_bets(probs_df), odds_map, bankroll)
    strict_best = apply_value_filter(strict_best)
    strict_best = apply_over15_quality_filter(strict_best)
    strict_best = apply_performance_filter(strict_best, history)
    strict_best = apply_league_weighting(strict_best, history)
    best_df = ensure_at_least_one(strict_best, soft_best)
    write_df(TAB_BEST_BETS, sort_by_date(best_df))

    # ---------- Combo_Bets ----------
    log("Combo_Bets...")
    soft_combos = build_combo_bets(probs_df, top_k=40)
    strict_combos = build_combo_bets(probs_df, top_k=COMBO_TOP_N)
    combos_df = ensure_at_least_one(strict_combos, soft_combos)
    write_df(TAB_COMBOS, sort_by_date(combos_df))

    # ---------- Today's Shortlist ----------
    shortlist = build_shortlist(best_df, safe_df, bal_df)
    if shortlist is None or shortlist.empty:
        shortlist = ensure_at_least_one(pd.DataFrame(), best_df, size=SHORTLIST_SIZE)
    write_df(TAB_SHORTLIST, sort_by_date(shortlist))

    # ---------- Dashboard ----------
    dashboard = build_dashboard(shortlist, history, bankroll)
    write_df(TAB_DASHBOARD, dashboard)

    # ---------- History ----------
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
