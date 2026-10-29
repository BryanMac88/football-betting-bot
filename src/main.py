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
    named_btts_prices, best_btts_prices,
)
from bet_tracking import combo_probs, COMBO_SPECS, evaluate_bet, decimal_to_fractional
from performance import apply_performance_filter, apply_league_weighting
from odds_sources import (
    get_fd_odds_cache, merge_odds_entry, model_implied_odds, attach_corner_avgs,
    lookup_match_corners,
)


# ================= CONFIG =================
# League focus (football-data.org codes):
#   England: PL, Championship, L1, L2, National League
#   Top EU: Ligue 1, La Liga, Bundesliga
#   Europe: Champions League, Europa League, Conference League
BASE_COMP_CODES = [
    "PL", "ELC", "EL1", "EL2", "ENL",
    "FL1", "PD", "BL1",
    "CL", "EL", "EC",
]
OPTIONAL_COMP_CODES: list = []

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
# Odds-bracket filter: drop brackets with poor historical ROI once sample is large enough
BRACKET_MIN_BETS = 20
BRACKET_MIN_ROI = -0.08          # unit ROI floor (e.g. -0.08 = -8% per bet)
BRACKET_FILTER_ENABLED = True
# Auto-update Config!B2 bankroll from settled stake-weighted P&L
AUTO_UPDATE_BANKROLL = True

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
TAB_ALL_GAMES = "All_Games_Best"
TAB_CORNERS_O95 = "Over_9_5_Corners"
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
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if s.endswith("%"):
                return float(s.rstrip("%").strip()) / 100.0
            x = s
        v = float(x)
        return default if math.isnan(v) else v
    except Exception:
        return default


def parse_rate(x: Any, default: float = float("nan")) -> float:
    """Parse 0.58, 58, or '58.00%' into a 0–1 rate."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if not s or s.lower() in ("nan", "none", "-"):
                return default
            if s.endswith("%"):
                return float(s.rstrip("%").strip()) / 100.0
            v = float(s)
        else:
            v = float(x)
        if math.isnan(v):
            return default
        if abs(v) > 1.5 and abs(v) <= 100:
            return v / 100.0
        return v
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
    if "CORNER" in b: return 0.90
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


# ================= EXPECTED CORNERS (single source: All_Games_Best) =================
# The All_Games_Best build is the source of truth for expected total corners.
# Its values are stamped onto every other tab (and the bet history) by match,
# so every tab shows the identical number for the same fixture.
_ALL_GAMES_CORNERS: Dict[Tuple[str, str, str], float] = {}


def set_corner_lookup(all_games_df: Optional[pd.DataFrame]) -> None:
    """Load expected total corners from the All_Games_Best frame, keyed by match."""
    _ALL_GAMES_CORNERS.clear()
    if all_games_df is None or all_games_df.empty or "exp_total_corners" not in all_games_df.columns:
        return
    for _, r in all_games_df.iterrows():
        v = pd.to_numeric(r.get("exp_total_corners"), errors="coerce")
        if pd.notna(v):
            key = (str(r.get("league", "")), str(r.get("home", "")), str(r.get("away", "")))
            _ALL_GAMES_CORNERS[key] = round(float(v), 2)


def stamp_corners(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Set exp_total_corners on any match-level frame from the All_Games_Best lookup."""
    if df is None or df.empty or not _ALL_GAMES_CORNERS:
        return df
    if not {"league", "home", "away"}.issubset(df.columns):
        return df
    out = df.copy()
    if "exp_total_corners" not in out.columns:
        out["exp_total_corners"] = ""
    out["exp_total_corners"] = out["exp_total_corners"].astype(object)
    old_vals = list(out["exp_total_corners"])
    new_vals = []
    for lg, h, a, old in zip(out["league"], out["home"], out["away"], old_vals):
        v = _ALL_GAMES_CORNERS.get((str(lg), str(h), str(a)))
        new_vals.append(v if v is not None else old)
    out["exp_total_corners"] = new_vals
    return out


def _move_corners_column(df: pd.DataFrame) -> pd.DataFrame:
    """Place exp_total_corners right after the bet column (or after away) so it is easy to see."""
    if df is None or df.empty or "exp_total_corners" not in df.columns:
        return df
    cols = list(df.columns)
    anchor = "bet" if "bet" in cols else ("away" if "away" in cols else None)
    if anchor is None:
        return df
    cols.remove("exp_total_corners")
    cols.insert(cols.index(anchor) + 1, "exp_total_corners")
    return df[cols]


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

# Columns that are rates in ~0–1 range and must display as percentages
PERCENT_COLUMNS = {
    "prob", "confidence", "edge", "league_avg", "league_avg_prob",
    "hit_rate", "avg_prob", "roi_per_bet", "win_rate", "recent_win_rate",
    "break_even", "kelly", "stake_pct", "max_stake_pct",
    "p_home", "p_draw", "p_away", "p_1x", "p_x2", "p_12",
    "p_btts_yes", "p_btts_no",
    "p_over_0_5", "p_over_1_5", "p_over_2_5", "p_over_3_5",
    "p_under_1_5", "p_under_2_5", "p_under_3_5",
}

NUMBER_COLUMNS = {
    "odds", "home_odds", "draw_odds", "away_odds", "home_xg", "away_xg",
    "league_base_total", "profit", "rank", "bets", "won", "lost",
    "n_home", "n_away", "hg", "ag", "fixtures_n", "results_n",
    "score", "adj", "penalty", "stake", "bankroll", "model_implied_odds",
    "avg_odds", "profit_money", "clv", "roi_stake_wtd",
    "home_corners_avg", "away_corners_avg",
    "exp_home_corners", "exp_away_corners", "exp_total_corners", "league_avg_corners",
    "actual_corners", "avg_exp_corners", "avg_actual_corners",
    "corners_bias", "corners_mae",
}

# Columns that get a yellow highlight on every tab they appear on
HIGHLIGHT_COLUMNS = {"exp_total_corners", "avg_exp_corners"}
HIGHLIGHT_RGB = {"red": 1.0, "green": 0.92, "blue": 0.55}


def _is_percent_col(name: str) -> bool:
    n = str(name).strip().lower()
    if n in PERCENT_COLUMNS:
        return True
    for token in ("prob", "rate", "pct", "percent", "confidence", "edge", "roi", "kelly"):
        if token in n:
            return True
    return False


def write_df(name: str, df: pd.DataFrame, corners_view: bool = True):
    """Write a DataFrame and auto-format probability columns as 0.00%.

    corners_view=True stamps expected total corners (from All_Games_Best) onto
    match-level frames and moves that column next to the bet. The highlight is
    applied whenever a highlight column is present. Bet_History passes False so
    the expected value logged at bet time is never overwritten.
    """
    sh = open_sheet()
    ws = _get_or_create_ws(sh, name)
    ws.clear()
    if df is None or df.empty:
        ws.update([["(no data)"]])
        return

    if corners_view:
        out = _move_corners_column(stamp_corners(df))
    else:
        out = df.copy()
    cols = list(out.columns)
    percent_idxs = [i for i, c in enumerate(cols) if _is_percent_col(c)]
    number_idxs = [i for i, c in enumerate(cols) if str(c).strip().lower() in NUMBER_COLUMNS]
    highlight_idxs = [i for i, c in enumerate(cols) if str(c).strip().lower() in HIGHLIGHT_COLUMNS]

    header = cols
    rows: List[List[Any]] = []
    for _, row in out.iterrows():
        cells: List[Any] = []
        for i, c in enumerate(cols):
            val = row[c]
            if pd.isna(val) or val is None or (isinstance(val, str) and str(val).strip() == ""):
                cells.append("")
                continue
            if i in percent_idxs or i in number_idxs:
                try:
                    num = float(val)
                    # If a % col was stored as 58 instead of 0.58, normalise
                    if i in percent_idxs and abs(num) > 1.5 and abs(num) <= 100:
                        num = num / 100.0
                    cells.append(num)
                except (TypeError, ValueError):
                    cells.append(str(val))
            else:
                cells.append(str(val))
        rows.append(cells)

    ws.update([header] + rows, value_input_option="USER_ENTERED")

    # Apply Sheets number formats so prob etc. show as 58.00%
    if not rows:
        return
    sheet_id = ws.id
    n_rows = len(rows)
    reqs = []
    for col_idx in percent_idxs:
        reqs.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": n_rows + 1,
                    "startColumnIndex": col_idx,
                    "endColumnIndex": col_idx + 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "numberFormat": {"type": "PERCENT", "pattern": "0.00%"}
                    }
                },
                "fields": "userEnteredFormat.numberFormat",
            }
        })
    for col_idx in number_idxs:
        col_name = str(cols[col_idx]).lower()
        pattern = "0" if col_name in (
            "rank", "bets", "won", "lost", "n_home", "n_away",
            "hg", "ag", "fixtures_n", "results_n",
        ) else "0.00"
        reqs.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": n_rows + 1,
                    "startColumnIndex": col_idx,
                    "endColumnIndex": col_idx + 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "numberFormat": {"type": "NUMBER", "pattern": pattern}
                    }
                },
                "fields": "userEnteredFormat.numberFormat",
            }
        })

    # Highlight expected-corners columns (header + data) in yellow, bold
    for col_idx in highlight_idxs:
        reqs.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": n_rows + 1,
                    "startColumnIndex": col_idx,
                    "endColumnIndex": col_idx + 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": HIGHLIGHT_RGB,
                        "textFormat": {"bold": True},
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
            }
        })
    # Dashboard-style tabs: highlight any metric row that is about corners
    if "metric" in cols:
        for pos, metric in enumerate(out["metric"].astype(str).tolist()):
            if "corner" in metric.lower():
                reqs.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": pos + 1,
                            "endRowIndex": pos + 2,
                            "startColumnIndex": 0,
                            "endColumnIndex": len(cols),
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": HIGHLIGHT_RGB,
                                "textFormat": {"bold": True},
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,textFormat)",
                    }
                })
    if reqs:
        try:
            sh.batch_update({"requests": reqs})
        except Exception as e:
            print(f"number format apply failed on '{name}': {e}", flush=True)

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


def set_bankroll(value: float) -> None:
    """Write current bankroll to Config!B2 (creates Config sheet if needed)."""
    try:
        sh = open_sheet()
        ws = _get_or_create_ws(sh, TAB_CONFIG, rows=20, cols=6)
        # Ensure labels exist
        existing = ws.get_all_values()
        if not existing or not existing[0]:
            ws.update("A1:B2", [["setting", "value"], ["bankroll", value]])
        else:
            ws.update("A2:B2", [["bankroll", value]])
        log(f"  bankroll updated on Config!B2 → {value:.2f}")
    except Exception as e:
        log(f"  bankroll write failed: {e}")


def apply_settled_to_bankroll(history, bankroll: float) -> float:
    """Add newly settled stake-weighted P&L into bankroll and persist.

    Uses profit_money when present; falls back to unit profit if stake was blank.
    Only rows with result WON/LOST and a numeric profit contribute.
    Tracks applied rows via a marker column `_bankroll_applied` stored in-sheet as blank/1
    is awkward — instead we only auto-adjust once per run using profits that look
    unsettled in money terms is fragile. Simpler approach: recompute bankroll as
    DEFAULT/starting + sum(all profit_money). That needs a starting bankroll cell.

    Practical approach used here:
      - Read Config!B3 as starting_bankroll (optional). If missing, use current B2 as baseline
        only when no settled profit_money exists yet.
      - New bankroll = starting + sum(profit_money of all settled bets with stake).
    """
    if not AUTO_UPDATE_BANKROLL:
        return bankroll
    if history is None or history.empty:
        return bankroll
    h = history.copy()
    settled = h[h["result"].astype(str).str.upper().isin(["WON", "LOST"])]
    if settled.empty:
        return bankroll
    money = pd.to_numeric(settled.get("profit_money"), errors="coerce")
    if money.notna().sum() == 0:
        # Fall back to unit profits (assume 1-unit stakes)
        money = pd.to_numeric(settled.get("profit"), errors="coerce")
    total_pnl = float(money.fillna(0).sum())

    # Starting bankroll: Config!B3 if set, else (current - total_pnl) to avoid double-count
    starting = None
    try:
        sh = open_sheet()
        ws = sh.worksheet(TAB_CONFIG)
        vals = ws.get_all_values()
        # Expect rows: [setting,value], [bankroll,X], [starting_bankroll,Y]
        for row in vals:
            if len(row) >= 2 and str(row[0]).strip().lower() in ("starting_bankroll", "starting", "start"):
                starting = float(row[1])
                break
    except Exception:
        starting = None

    if starting is None:
        # Derive: treat current bankroll as already including past P&L only if we can't find start
        # Safer: set starting = bankroll - total_pnl so new = bankroll (no jump), then next
        # settles move it. On first enable this keeps bankroll stable.
        starting = bankroll - total_pnl
        try:
            sh = open_sheet()
            ws = _get_or_create_ws(sh, TAB_CONFIG, rows=20, cols=6)
            ws.update("A3:B3", [["starting_bankroll", round(starting, 2)]])
        except Exception as e:
            log(f"  starting_bankroll seed failed: {e}")

    new_bankroll = round(starting + total_pnl, 2)
    if abs(new_bankroll - bankroll) >= 0.01:
        set_bankroll(new_bankroll)
        log(f"  bankroll {bankroll:.2f} → {new_bankroll:.2f} (P&L {total_pnl:+.2f})")
        return new_bankroll
    return bankroll


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
    """Pull live bookmaker prices from The Odds API (h2h + totals + BTTS)."""
    api = OddsApi(env("ODDS_API_KEY"))
    events_by_sport = {}
    matched = {}
    for league, sub in fx_df.groupby("league"):
        sport_key = FD_TO_ODDS_SPORT.get(league)
        if not sport_key:
            continue
        if sport_key not in events_by_sport:
            try:
                # Live bookmaker prices: 1X2, totals, BTTS
                events_by_sport[sport_key] = api.get_odds(
                    sport_key, markets=["h2h", "totals", "btts"]
                )
                log(f"  odds API {sport_key}: {len(events_by_sport[sport_key])} events")
            except Exception as e:
                log(f"odds fetch failed for {sport_key}: {e}")
                events_by_sport[sport_key] = []
        for _, r in sub.iterrows():
            ev = match_fixture(r["home"], r["away"], r["utcDate"], events_by_sport[sport_key])
            if ev is None:
                continue
            named_h2h, book_h2h = named_h2h_prices(ev)
            named_tot, book_tot = named_totals_prices(ev)
            named_btts, book_btts = named_btts_prices(ev)
            matched[(league, r["home"], r["away"])] = {
                "h2h": best_h2h_prices(ev),
                "named_h2h": named_h2h,
                "named_totals": named_tot,
                "all_totals": all_totals_prices(ev),
                "named_btts": named_btts,
                "btts": best_btts_prices(ev),
                "book": book_h2h or book_tot or book_btts,
                "event": ev,
            }
    # Optional Scrapling fallback for fixtures still missing preferred-book prices
    try:
        from scrapling_odds import enrich_named_odds, _normalize as _odds_norm
        missing = []
        for _, r in fx_df.iterrows():
            key = (r.get("league"), r.get("home"), r.get("away"))
            entry = matched.get(key)
            if entry is None or not entry.get("named_h2h"):
                missing.append({"league": r.get("league"), "home": r.get("home"), "away": r.get("away")})
        if missing:
            scraped = enrich_named_odds(missing)
            filled = 0
            for item in missing:
                nk = (_odds_norm(item["home"]), _odds_norm(item["away"]))
                row = scraped.get(nk)
                if not row:
                    continue
                key = (item["league"], item["home"], item["away"])
                entry = matched.get(key) or {
                    "h2h": {}, "named_h2h": {}, "named_totals": {},
                    "all_totals": {}, "named_btts": {}, "btts": {},
                    "book": None, "event": None,
                }
                entry["named_h2h"] = {
                    item["home"]: row["home"],
                    "Draw": row["draw"],
                    item["away"]: row["away"],
                }
                entry["book"] = row.get("book", "paddypower")
                entry["h2h"] = entry.get("h2h") or entry["named_h2h"]
                matched[key] = entry
                filled += 1
            if filled:
                log(f"  scrapling filled named odds for {filled} fixture(s)")
    except Exception as e:
        log(f"scrapling odds enrichment skipped: {e}")

    # ---- football-data.co.uk (Bet365 / market avg) — free, no key ----
    try:
        fd_cache = get_fd_odds_cache()
        fd_cache.load_for_fixtures(list(fx_df["league"].unique()) if "league" in fx_df.columns else [])
        filled_fd = 0
        for _, r in fx_df.iterrows():
            key = (r.get("league"), r.get("home"), r.get("away"))
            fd_entry = fd_cache.lookup(r.get("league", ""), r.get("home", ""), r.get("away", ""))
            if not fd_entry:
                continue
            if key in matched:
                matched[key] = merge_odds_entry(matched[key], fd_entry)
            else:
                matched[key] = fd_entry
            filled_fd += 1
        if filled_fd:
            log(f"  football-data.co.uk odds for {filled_fd} fixture(s)")
    except Exception as e:
        log(f"football-data.co.uk odds skipped: {e}")

    with_named = sum(1 for v in matched.values() if v.get("named_h2h") or v.get("named_totals") or v.get("named_btts") or v.get("h2h"))
    log(f"odds matched: {len(matched)} fixtures ({with_named} with bookmaker prices)")
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
    """Resolve a real bookmaker decimal price for a pick."""
    if not odds_entry:
        return None, "none"
    b = bet.strip().upper()
    named_h2h = odds_entry.get("named_h2h", {})
    named_tot = odds_entry.get("named_totals", {})
    named_btts = odds_entry.get("named_btts", {})
    all_h2h = odds_entry.get("h2h", {})
    all_tot = odds_entry.get("all_totals", {})
    all_btts = odds_entry.get("btts", {})
    book = odds_entry.get("book")

    if b in ("HOME WIN", "AWAY WIN", "DRAW"):
        key = home if b == "HOME WIN" else (away if b == "AWAY WIN" else "Draw")
        if key in named_h2h:
            return named_h2h[key], book or "named"
        if key in all_h2h:
            return all_h2h[key], "market-best"
        return None, "none"

    if b in ("BTTS YES", "BTTS NO"):
        side = "Yes" if b == "BTTS YES" else "No"
        if side in named_btts:
            return named_btts[side], book or "named"
        if side in all_btts:
            return all_btts[side], "market-best"
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

    # Double chance — approximate from 1X2 when no dedicated market
    if b.startswith("DOUBLE CHANCE") and "&" not in b:
        if "1X" in b and home in all_h2h and "Draw" in all_h2h:
            p = (1.0 / all_h2h[home]) + (1.0 / all_h2h["Draw"])
            return round(1.0 / p, 3) if p > 0 else None, "approx-1x2"
        if "X2" in b and away in all_h2h and "Draw" in all_h2h:
            p = (1.0 / all_h2h[away]) + (1.0 / all_h2h["Draw"])
            return round(1.0 / p, 3) if p > 0 else None, "approx-1x2"
        if b.endswith("12") and home in all_h2h and away in all_h2h:
            p = (1.0 / all_h2h[home]) + (1.0 / all_h2h[away])
            return round(1.0 / p, 3) if p > 0 else None, "approx-1x2"
        return None, "none"

    # Combined bets (e.g. "HOME WIN & OVER 1.5"): multiply single-leg prices.
    # Not a true book accumulator price (correlation / bet-builder margin differ),
    # but far better than blank / pure model-implied for tracking.
    if "&" in b:
        legs = [x.strip() for x in bet.split("&") if x.strip()]
        if len(legs) >= 2:
            prices, sources = [], []
            for leg in legs:
                lp, ls = odds_for_bet(odds_entry, leg, home, away)
                if lp is None or float(lp) <= 1.0:
                    return None, "none"
                prices.append(float(lp))
                sources.append(ls)
            # Mild correlation haircut so we do not overstate price on related legs
            raw = 1.0
            for p in prices:
                raw *= p
            combined = round(raw * 0.97, 3)  # ~3% builder-style haircut
            src = "combo-approx"
            if all(s not in ("none", "model-implied", None) for s in sources):
                src = "combo-from-legs"
            return combined, src

    return None, "none"

def passes_odds_filter(price, source):
    if price is None:
        return not ODDS_FILTER_REQUIRE_NAMED_BOOK
    p = float(price)
    return MIN_DECIMAL_ODDS <= p <= MAX_DECIMAL_ODDS

def attach_odds(df, odds_map, bankroll, apply_filter=True):
    """Attach real bookmaker prices; fall back to model-implied so odds never blank.

    Also computes market edge = model_prob - 1/odds (positive = value vs the price).
    """
    if df is None or df.empty:
        return df
    prices, fracs, sources, keeps, stakes, edges = [], [], [], [], [], []
    for _, r in df.iterrows():
        entry = odds_map.get((r.get("league"), r.get("home"), r.get("away")))
        price, source = odds_for_bet(entry, str(r.get("bet", "")), r.get("home"), r.get("away"))
        # Guaranteed fallback: model-implied odds from probability
        if price is None:
            price = model_implied_odds(r.get("prob"))
            if price is not None:
                source = "model-implied"
        prices.append(round(price, 3) if price else "")
        fracs.append(decimal_to_fractional(price) if price else "")
        sources.append(source or "none")
        keeps.append(passes_odds_filter(price, source))
        stakes.append(kelly_stake(r.get("prob"), price, bankroll) if price else "")
        edge_val = ""
        try:
            p = float(r.get("prob"))
            o = float(price) if price else None
            if o and o > 1.0 and 0.0 < p < 1.0:
                edge_val = round(p - (1.0 / o), 4)
        except (TypeError, ValueError):
            edge_val = ""
        edges.append(edge_val)
    out = df.copy()
    out["odds"] = prices
    out["fractional"] = fracs
    out["odds_source"] = sources
    out["stake"] = stakes
    out["edge"] = edges
    if "closing_odds" not in out.columns:
        out["closing_odds"] = ""
    if apply_filter:
        before = len(out)
        out = out[pd.Series(keeps, index=out.index)].reset_index(drop=True)
        if before - len(out):
            log(f"  odds filter removed {before - len(out)} pick(s)")
        if "rank" in out.columns and not out.empty:
            out["rank"] = range(1, len(out) + 1)
    n_real = sum(1 for s in sources if s and s not in ("none", "model-implied", "approx-1x2"))
    n_model = sum(1 for s in sources if s == "model-implied")
    log(f"  odds attached: {n_real} bookmaker, {n_model} model-implied, {len(prices)} total")
    return out

def calculate_odds_bracket_roi(history: pd.DataFrame, min_bets: int = BRACKET_MIN_BETS) -> Dict[str, float]:
    """Return {odds_bracket: unit_roi} for brackets with enough settled bets."""
    if history is None or history.empty:
        return {}
    h = history.copy()
    h["result"] = h["result"].astype(str).str.upper()
    settled = h[h["result"].isin(["WON", "LOST"])]
    if settled.empty or "odds" not in settled.columns:
        return {}
    settled = settled.copy()
    settled["odds_num"] = pd.to_numeric(settled["odds"], errors="coerce")
    settled["profit_num"] = pd.to_numeric(settled.get("profit"), errors="coerce")
    settled["bracket"] = settled["odds_num"].map(_odds_bracket)
    out: Dict[str, float] = {}
    for br, sub in settled.groupby("bracket"):
        if br == "unknown / missing" or len(sub) < min_bets:
            continue
        roi = sub["profit_num"].mean()
        if roi == roi:
            out[str(br)] = float(roi)
    return out


def apply_odds_bracket_filter(
    picks: pd.DataFrame,
    history: pd.DataFrame,
    min_bets: int = BRACKET_MIN_BETS,
    min_roi: float = BRACKET_MIN_ROI,
) -> pd.DataFrame:
    """Drop picks whose decimal odds fall in historically losing brackets."""
    if not BRACKET_FILTER_ENABLED:
        return picks
    if picks is None or picks.empty or "odds" not in picks.columns:
        return picks
    bracket_roi = calculate_odds_bracket_roi(history, min_bets=min_bets)
    if not bracket_roi:
        return picks
    bad = {br for br, roi in bracket_roi.items() if roi < min_roi}
    if not bad:
        return picks

    out = picks.copy()
    odds_num = pd.to_numeric(out["odds"], errors="coerce")
    brackets = odds_num.map(_odds_bracket)
    keep = ~brackets.isin(bad)
    removed = int((~keep).sum())
    if removed:
        log(f"  odds-bracket filter removed {removed} pick(s) in {sorted(bad)}")
        out = out[keep].copy()
        if "rank" in out.columns and not out.empty:
            out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


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
        out = _dedupe_matches(strict_df.copy())
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

    out = _dedupe_matches(out, limit=size)
    out["warning"] = "BELOW ACCEPTABLE LEVEL – bet at your own risk"
    if "rank" in out.columns:
        out["rank"] = range(1, len(out) + 1)
    else:
        out.insert(0, "rank", range(1, len(out) + 1))
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
                          "score": round(sc, 4),
                          "home_corners_avg": r.get("home_corners_avg", ""),
                          "away_corners_avg": r.get("away_corners_avg", ""),
                          "exp_total_corners": r.get("exp_total_corners", ""),
                          "league_avg_corners": r.get("league_avg_corners", "")})
    df = pd.DataFrame(picks)
    if df.empty:
        return df
    df = df.sort_values(["score", "edge", "prob"], ascending=[False, False, False]).reset_index(drop=True)
    # Always one bet per match first
    df = _dedupe_matches(df, limit=None)
    if unique_teams:
        df = _dedupe_teams(df, max_rows)
    else:
        df = df.head(max_rows).reset_index(drop=True)
    if "rank" in df.columns:
        df["rank"] = range(1, len(df) + 1)
    else:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df

CORE_COLS = ["utcDate", "league", "home", "away"]

def _match_key(row) -> Tuple[str, str, str]:
    """Identity of a fixture — used to enforce one bet per game."""
    return (
        str(row.get("utcDate", ""))[:16],
        str(row.get("home", "")).strip().lower(),
        str(row.get("away", "")).strip().lower(),
    )


def _dedupe_matches(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Keep only ONE row per fixture (best already-sorted first).
    Prevents the same game appearing with multiple different bets on a tab.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    # Prefer higher score / prob if not already sorted
    out = df.copy()
    sort_cols = []
    if "score" in out.columns:
        sort_cols.append("score")
    if "prob" in out.columns:
        sort_cols.append("prob")
    if "edge" in out.columns:
        sort_cols.append("edge")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    seen, kept = set(), []
    for _, row in out.iterrows():
        key = _match_key(row)
        if key in seen:
            continue
        if not key[1] and not key[2]:
            continue
        seen.add(key)
        kept.append(row)
        if limit is not None and len(kept) >= limit:
            break

    if not kept:
        return out.head(0)

    result = pd.DataFrame(kept).reset_index(drop=True)
    if "rank" in result.columns:
        result["rank"] = range(1, len(result) + 1)
    else:
        result.insert(0, "rank", range(1, len(result) + 1))
    return result


def _dedupe_teams(df, limit):
    """One pick per team (stricter than match-level — used for Top10 lists)."""
    used, kept = set(), []
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
    return pd.DataFrame(kept).reset_index(drop=True) if kept else df.head(0)

def top_n_for_market(probs_df, prob_col, bet_label, top_n=TOP_N):
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()
    extra = [c for c in (
        "home_corners_avg", "away_corners_avg", "exp_total_corners", "league_avg_corners"
    ) if c in probs_df.columns]
    df = probs_df[CORE_COLS + ["confidence", "home_xg", "away_xg", "league_base_total", prob_col] + extra].copy()
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
    keep = ["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score", "adj", "penalty"] + extra
    return df[[c for c in keep if c in df.columns]]

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
                     "home_xg": r.get("home_xg"), "away_xg": r.get("away_xg"),
                     "home_corners_avg": r.get("home_corners_avg", ""),
                     "away_corners_avg": r.get("away_corners_avg", ""),
                     "exp_total_corners": r.get("exp_total_corners", ""),
                     "league_avg_corners": r.get("league_avg_corners", "")})
    df = pd.DataFrame(rows).sort_values(["prob", "confidence"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def build_all_games_best(probs_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fixture: the single most probable outcome.
    NO filters — every available game is included.
    Sorted by league first, then kick-off time.
    """
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    # Prefer 1X2 + goals + BTTS as the "main" outcomes for a full card
    markets = [
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
        ("p_12", "DOUBLE CHANCE 12"),
    ]

    rows = []
    for _, r in probs_df.iterrows():
        best_mkt, best_p = None, -1.0
        for col, label in markets:
            try:
                p = float(r.get(col, 0) or 0)
            except (TypeError, ValueError):
                p = 0.0
            if p > best_p:
                best_p, best_mkt = p, label
        if best_mkt is None:
            continue
        rows.append({
            "league": r.get("league", ""),
            "utcDate": r.get("utcDate", ""),
            "home": r.get("home", ""),
            "away": r.get("away", ""),
            "bet": best_mkt,
            "prob": round(best_p, 4),
            "confidence": r.get("confidence", ""),
            "home_xg": r.get("home_xg", ""),
            "away_xg": r.get("away_xg", ""),
            "home_corners_avg": r.get("home_corners_avg", ""),
            "away_corners_avg": r.get("away_corners_avg", ""),
            "exp_home_corners": r.get("exp_home_corners", ""),
            "exp_away_corners": r.get("exp_away_corners", ""),
            "exp_total_corners": r.get("exp_total_corners", ""),
            "league_avg_corners": r.get("league_avg_corners", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort: league first, then kick-off
    df["_dt"] = pd.to_datetime(df["utcDate"], errors="coerce", utc=True)
    df = df.sort_values(["league", "_dt"], ascending=[True, True]).drop(columns=["_dt"])
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def _poisson_cdf_ge(lam: float, k: int) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    # P(X >= k) = 1 - P(X <= k-1)
    s = 0.0
    term = math.exp(-lam)
    for i in range(0, k):
        if i > 0:
            term *= lam / i
        s += term
    return max(0.0, min(1.0, 1.0 - s))


def build_over_corners(
    probs_df: pd.DataFrame,
    line: float = 9.5,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Rank fixtures by P(corners > line) using Poisson(exp_total_corners)."""
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()
    if "exp_total_corners" not in probs_df.columns:
        return pd.DataFrame()

    bet_label = f"OVER {line} CORNERS"
    rows = []
    for _, r in probs_df.iterrows():
        try:
            lam = float(r.get("exp_total_corners") or 0)
        except (TypeError, ValueError):
            continue
        if lam <= 0:
            continue
        # Over 9.5 → need 10+ corners
        need = int(math.floor(line) + 1)
        prob = _poisson_cdf_ge(lam, need)
        if prob < 0.35:
            continue
        conf = safe_float(r.get("confidence"), 0.5)
        rows.append({
            "utcDate": r.get("utcDate", ""),
            "league": r.get("league", ""),
            "home": r.get("home", ""),
            "away": r.get("away", ""),
            "bet": bet_label,
            "prob": round(prob, 3),
            "confidence": round(conf, 3),
            "exp_total_corners": round(lam, 2),
            "home_corners_avg": r.get("home_corners_avg", ""),
            "away_corners_avg": r.get("away_corners_avg", ""),
            "league_avg_corners": r.get("league_avg_corners", ""),
            "score": round(prob * conf, 4),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["score", "prob", "exp_total_corners"],
                                        ascending=[False, False, False]).reset_index(drop=True)
    df = _dedupe_matches(df, limit=top_n)
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    else:
        df["rank"] = range(1, len(df) + 1)
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
                         "score": round(sc, 4),
                         "exp_total_corners": r.get("exp_total_corners", ""),
                         "home_corners_avg": r.get("home_corners_avg", ""),
                         "away_corners_avg": r.get("away_corners_avg", "")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["score", "prob"], ascending=[False, False]).reset_index(drop=True)
    # One combo per match only (best scoring combo wins)
    df = _dedupe_matches(df, limit=top_k)
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

    # Drop existing rank column if present
    if "rank" in mix.columns:
        mix = mix.drop(columns=["rank"])

    if "score" in mix.columns:
        mix = mix.sort_values("score", ascending=False)
    elif "prob" in mix.columns:
        mix = mix.sort_values("prob", ascending=False)

    # One bet per match — never the same game twice with different markets
    mix = _dedupe_matches(mix, limit=size)

    keep = ["rank", "utcDate", "league", "home", "away", "bet", "prob", "odds", "fractional",
            "stake", "edge", "confidence", "score", "source", "odds_source", "warning",
            "exp_total_corners", "home_corners_avg", "away_corners_avg"]
    return mix[[c for c in keep if c in mix.columns]]

def build_dashboard(shortlist, history, bankroll):
    rows = []
    rows.append({"metric": "Run time (UTC)", "value": now().strftime("%Y-%m-%d %H:%M")})
    rows.append({"metric": "Bankroll", "value": f"{bankroll:.0f}"})
    rows.append({"metric": "Shortlist size", "value": len(shortlist) if shortlist is not None else 0})
    if _ALL_GAMES_CORNERS:
        corner_vals = list(_ALL_GAMES_CORNERS.values())
        rows.append({"metric": "Avg expected total corners (all games)",
                     "value": f"{sum(corner_vals) / len(corner_vals):.2f}"})
    if shortlist is not None and not shortlist.empty and "stake" in shortlist.columns:
        total_stake = pd.to_numeric(shortlist["stake"], errors="coerce").sum()
        rows.append({"metric": "Total recommended stake", "value": f"{total_stake:.2f}"})
        rows.append({"metric": "% of bankroll", "value": f"{(total_stake / bankroll * 100):.1f}%"})
        if "edge" in shortlist.columns:
            avg_edge = pd.to_numeric(shortlist["edge"], errors="coerce").mean()
            if avg_edge == avg_edge:
                rows.append({"metric": "Shortlist avg edge", "value": f"{avg_edge * 100:.1f}%"})
    if history is not None and not history.empty:
        res_u = history["result"].astype(str).str.upper()
        pending = int((res_u == "PENDING").sum())
        settled = history[res_u.isin(["WON", "LOST"])]
        rows.append({"metric": "Pending bets", "value": pending})
        if not settled.empty:
            wins = int((settled["result"].astype(str).str.upper() == "WON").sum())
            n = len(settled)
            rows.append({"metric": "Settled bets", "value": n})
            rows.append({"metric": "Historical hit rate", "value": f"{wins / n * 100:.0f}%"})
            profit = pd.to_numeric(settled.get("profit"), errors="coerce")
            rows.append({"metric": "Historical profit (units)", "value": f"{profit.sum():.2f}"})
            rows.append({"metric": "ROI per bet (units)", "value": f"{profit.mean():.3f}"})
            money = pd.to_numeric(settled.get("profit_money"), errors="coerce")
            stakes = pd.to_numeric(settled.get("stake"), errors="coerce")
            if money.notna().any() and stakes.fillna(0).sum() > 0:
                rows.append({"metric": "ROI stake-weighted", "value": f"{money.fillna(0).sum() / stakes.fillna(0).sum():.3f}"})
            clv = pd.to_numeric(settled.get("clv"), errors="coerce")
            if clv.notna().any():
                rows.append({"metric": "Avg CLV (prob pts)", "value": f"{clv.mean():+.4f}"})
            odds_num = pd.to_numeric(settled.get("odds"), errors="coerce")
            brackets = odds_num.map(_odds_bracket)
            best_br, best_roi = None, None
            for br, idx in brackets.groupby(brackets).groups.items():
                if br == "unknown / missing" or len(idx) < 5:
                    continue
                r = profit.loc[idx].mean()
                if r == r and (best_roi is None or r > best_roi):
                    best_br, best_roi = br, r
            if best_br is not None:
                rows.append({"metric": "Best odds bracket (ROI)", "value": f"{best_br} → {best_roi:+.3f}"})
    return pd.DataFrame(rows)


# ================= HISTORY / ACCURACY =================
HISTORY_COLS = [
    "logged_utc", "utcDate", "league", "home", "away", "bet", "source",
    "prob", "odds", "fractional", "odds_source", "stake", "edge", "closing_odds",
    "result", "score", "profit", "profit_money", "clv",
    "exp_total_corners", "actual_corners",
]

def _hist_key(row):
    """Full key including source — allows same match in multiple tabs if desired."""
    return (str(row.get("utcDate", ""))[:10], str(row.get("home", "")),
            str(row.get("away", "")), str(row.get("bet", "")), str(row.get("source", "")))


def _match_bet_key(row):
    """Match + market only (no source) — used to avoid duplicate tracking rows."""
    return (str(row.get("utcDate", ""))[:10], str(row.get("home", "")),
            str(row.get("away", "")), str(row.get("bet", "")).upper())


# Tabs that are allowed to create a *new* history row even if another tab
# already logged the same match+bet. Primary recommendation tabs only.
_PRIMARY_HISTORY_SOURCES = {
    "Best_Bets", "Safe_Picks", "Balanced_Picks", "Combo_Bets",
    "Over_9_5_Corners",
}


def append_new_picks(history, picks, source):
    """Append picks to Bet_History.

    Primary sources (Best/Safe/Balanced/Combo) always log if that source has not
    logged the same match+bet yet.
    Secondary sources (Top10_*) only log when *no* source has logged that match+bet,
    so Accuracy is not flooded with duplicates.

    Expected total corners (from All_Games_Best) is stored with every logged bet
    so the Accuracy tab can compare it with the actual corners later.
    """
    if picks is None or picks.empty:
        return history
    picks = stamp_corners(picks)
    existing_full = set()
    existing_match_bet = set()
    if history is not None and not history.empty:
        for _, h in history.iterrows():
            existing_full.add(_hist_key(h))
            existing_match_bet.add(_match_bet_key(h))

    new_rows = []
    stamp = now().strftime("%Y-%m-%d %H:%M:%S UTC")
    primary = source in _PRIMARY_HISTORY_SOURCES
    for _, p in picks.iterrows():
        full = (str(p.get("utcDate", ""))[:10], str(p.get("home", "")),
                str(p.get("away", "")), str(p.get("bet", "")), str(source))
        mb = (str(p.get("utcDate", ""))[:10], str(p.get("home", "")),
              str(p.get("away", "")), str(p.get("bet", "")).upper())
        if full in existing_full:
            continue
        if not primary and mb in existing_match_bet:
            continue
        new_rows.append({
            "logged_utc": stamp, "utcDate": p.get("utcDate", ""), "league": p.get("league", ""),
            "home": p.get("home", ""), "away": p.get("away", ""), "bet": p.get("bet", ""),
            "source": source, "prob": p.get("prob", ""), "odds": p.get("odds", ""),
            "fractional": p.get("fractional", ""), "odds_source": p.get("odds_source", ""),
            "stake": p.get("stake", ""), "edge": p.get("edge", ""),
            "closing_odds": p.get("closing_odds", ""), "result": "PENDING",
            "score": "", "profit": "", "profit_money": "", "clv": "",
            "exp_total_corners": p.get("exp_total_corners", ""), "actual_corners": "",
        })
        existing_full.add(full)
        existing_match_bet.add(mb)
    if not new_rows:
        return history
    log(f"  logging {len(new_rows)} new pick(s) from {source}")
    add = pd.DataFrame(new_rows)
    return add if history is None or history.empty else pd.concat([history, add], ignore_index=True)

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
    for col in ("result", "score", "profit", "profit_money", "clv",
                "exp_total_corners", "actual_corners"):
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
        bet_str = str(h.get("bet", ""))
        total_corners = None
        if "CORNER" in bet_str.upper():
            total_corners = lookup_match_corners(
                str(h.get("league", "")),
                str(h.get("home", "")),
                str(h.get("away", "")),
                str(h.get("utcDate", "")),
            )
            if total_corners is None:
                continue  # keep PENDING until corners appear in CSV
        won = evaluate_bet(bet_str, hg, ag, total_corners=total_corners)
        if won is None:
            continue
        out.at[i, "result"] = "WON" if won else "LOST"
        out.at[i, "score"] = f"{hg}-{ag}"
        # Actual total corners for the expected-corners accuracy (best effort, never blocks settling)
        actual_corners = total_corners
        if actual_corners is None:
            try:
                actual_corners = lookup_match_corners(
                    str(h.get("league", "")),
                    str(h.get("home", "")),
                    str(h.get("away", "")),
                    str(h.get("utcDate", "")),
                )
            except Exception:
                actual_corners = None
        out.at[i, "actual_corners"] = actual_corners if actual_corners is not None else ""
        try:
            odds = float(h.get("odds") or 0)
        except Exception:
            odds = 0.0
        try:
            stake = float(h.get("stake") or 0)
        except Exception:
            stake = 0.0
        # Unit profit (1-unit stake) and money profit (Kelly stake)
        if odds > 1:
            unit = round(odds - 1, 3) if won else -1.0
            out.at[i, "profit"] = unit
            out.at[i, "profit_money"] = round(stake * unit, 2) if stake > 0 else ""
        else:
            out.at[i, "profit"] = ""
            out.at[i, "profit_money"] = ""
        # Closing-line value: positive CLV means you beat the close (higher entry price)
        try:
            closing = float(h.get("closing_odds") or 0)
        except Exception:
            closing = 0.0
        if odds > 1 and closing > 1:
            # CLV in probability points: implied_close - implied_entry
            out.at[i, "clv"] = round((1.0 / closing) - (1.0 / odds), 4)
        else:
            out.at[i, "clv"] = ""
        settled += 1
    if settled:
        log(f"  settled {settled} bet(s)")

    # Backfill actual corners for already-settled bets that were logged with an
    # expected-corners value but whose corners were not yet in the CSV at settle time.
    backfilled = 0
    for i, h in out.iterrows():
        if str(h.get("result", "")).upper() not in ("WON", "LOST"):
            continue
        if str(h.get("actual_corners", "")).strip() != "":
            continue
        if str(h.get("exp_total_corners", "")).strip() == "":
            continue
        try:
            got = lookup_match_corners(
                str(h.get("league", "")),
                str(h.get("home", "")),
                str(h.get("away", "")),
                str(h.get("utcDate", "")),
            )
        except Exception:
            got = None
        if got is None:
            continue
        out.at[i, "actual_corners"] = got
        backfilled += 1
    if backfilled:
        log(f"  backfilled actual corners on {backfilled} settled bet(s)")
    return out

def refresh_closing_odds(history, odds_map):
    """For still-PENDING bets, refresh closing_odds from the latest odds snapshot.

    Practical CLV proxy: entry odds (logged) vs latest pre-kick price.
    Finished matches drop out of odds_map and are left unchanged.
    """
    if history is None or history.empty or not odds_map:
        return history
    out = history.copy()
    if "closing_odds" not in out.columns:
        out["closing_odds"] = ""
    out["closing_odds"] = out["closing_odds"].astype(object)
    updated = 0
    for i, h in out.iterrows():
        if str(h.get("result", "")).upper() not in ("", "PENDING"):
            continue
        key = (h.get("league"), h.get("home"), h.get("away"))
        entry = odds_map.get(key)
        if not entry:
            continue
        price, _src = odds_for_bet(entry, str(h.get("bet", "")), h.get("home"), h.get("away"))
        if price is None:
            continue
        try:
            price = round(float(price), 3)
        except (TypeError, ValueError):
            continue
        if str(h.get("closing_odds", "")).strip() == str(price):
            continue
        out.at[i, "closing_odds"] = price
        updated += 1
    if updated:
        log(f"  refreshed closing odds on {updated} pending bet(s)")
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

def _odds_bracket(odds_val) -> str:
    """Bucket decimal odds for accuracy tracking."""
    try:
        o = float(odds_val)
    except (TypeError, ValueError):
        return "unknown / missing"
    if o != o or o <= 1.0:  # NaN or invalid
        return "unknown / missing"
    if o < 1.50:
        return "1.01 – 1.49 (heavy fav)"
    if o < 1.80:
        return "1.50 – 1.79 (fav)"
    if o < 2.20:
        return "1.80 – 2.19 (near even)"
    if o < 2.80:
        return "2.20 – 2.79 (mild dog)"
    if o < 3.50:
        return "2.80 – 3.49 (dog)"
    if o < 5.00:
        return "3.50 – 4.99 (longshot)"
    return "5.00+ (very long)"


def _corners_accuracy_rows(settled: pd.DataFrame) -> List[Dict[str, Any]]:
    """Expected-total-corners accuracy: expected (from All_Games_Best, logged at bet time)
    vs actual total corners, one row per match (not per bet)."""
    placeholder = [{
        "grouping": "expected corners",
        "value": "(waiting for settled matches with corners data)",
        "bets": 0, "note": "",
    }]
    if settled is None or settled.empty:
        return placeholder
    if "exp_total_corners" not in settled.columns or "actual_corners" not in settled.columns:
        return placeholder

    c = settled.copy()
    c["exp_c"] = pd.to_numeric(c["exp_total_corners"], errors="coerce")
    c["act_c"] = pd.to_numeric(c["actual_corners"], errors="coerce")
    c = c.dropna(subset=["exp_c", "act_c"])
    if c.empty:
        return placeholder
    c["_match"] = (c["utcDate"].astype(str).str[:10] + "|" +
                   c["home"].astype(str) + "|" + c["away"].astype(str))
    c = c.drop_duplicates(subset=["_match"], keep="first")
    c["err"] = c["exp_c"] - c["act_c"]

    def _row(value, sub):
        n = len(sub)
        within2 = float((sub["err"].abs() <= 2).mean())
        return {
            "grouping": "expected corners",
            "value": value,
            "bets": n,
            "avg_exp_corners": round(float(sub["exp_c"].mean()), 2),
            "avg_actual_corners": round(float(sub["act_c"].mean()), 2),
            "corners_bias": round(float(sub["err"].mean()), 2),
            "corners_mae": round(float(sub["err"].abs().mean()), 2),
            "corners_within_2": f"{round(within2 * 100)}%",
            "note": "LOW SAMPLE" if n < SAMPLE_SIZE_WARNING else "",
        }

    rows = [_row("all matches", c)]
    if "league" in c.columns:
        for lg, sub in c.groupby("league"):
            if str(lg).strip():
                rows.append(_row(f"league: {lg}", sub))
    return rows


def build_accuracy(history):
    """Settled-bet performance by overall / bet / source / league / odds bracket / odds source.

    OVERALL and time-window rows de-dupe the same match+bet logged from multiple tabs
    (Best/Safe/Balanced/Top10) so hit rate is not inflated.
    Includes unit ROI, stake-weighted ROI, and avg CLV when available.
    Also includes an 'expected corners' block: expected total corners (from
    All_Games_Best) vs actual total corners.
    """
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    h["result"] = h["result"].astype(str).str.upper()
    settled = h[h["result"].isin(["WON", "LOST"])].copy()
    if settled.empty:
        return pd.DataFrame([{
            "grouping": "(nothing settled yet)", "value": "", "bets": 0,
            "won": 0, "lost": 0, "hit_rate": "", "avg_prob": "",
            "avg_odds": "", "break_even": "", "roi_per_bet": "",
            "roi_stake_wtd": "", "avg_clv": "", "note": "",
        }])

    # Prefer primary tabs when the same match+market was logged more than once
    _src_rank = {
        "Best_Bets": 0, "Safe_Picks": 1, "Balanced_Picks": 2,
        "Todays_Shortlist": 3, "Combo_Bets": 4,
        "Top10_Over_1_5": 5, "Top10_Over_2_5": 6,
    }
    settled["_src_rank"] = settled.get("source", pd.Series([""] * len(settled))).map(
        lambda s: _src_rank.get(str(s), 9)
    )
    settled["_match_bet"] = (
        settled["utcDate"].astype(str).str[:10] + "|" +
        settled["home"].astype(str) + "|" +
        settled["away"].astype(str) + "|" +
        settled["bet"].astype(str).str.upper()
    )
    settled = settled.sort_values("_src_rank")
    settled_dedup = settled.drop_duplicates(subset=["_match_bet"], keep="first").copy()

    def _prep(df):
        out = df.copy()
        out["won_flag"] = (out["result"] == "WON").astype(int)
        out["prob_num"] = out["prob"].map(lambda x: parse_rate(x))
        out["profit_num"] = pd.to_numeric(out.get("profit"), errors="coerce")
        out["profit_money_num"] = pd.to_numeric(out.get("profit_money"), errors="coerce")
        out["stake_num"] = pd.to_numeric(out.get("stake"), errors="coerce")
        out["odds_num"] = pd.to_numeric(out.get("odds"), errors="coerce")
        out["clv_num"] = pd.to_numeric(out.get("clv"), errors="coerce")
        out["odds_bracket"] = out["odds_num"].map(_odds_bracket)
        if "odds_source" not in out.columns:
            out["odds_source"] = ""
        if "league" not in out.columns:
            out["league"] = ""
        return out

    settled = _prep(settled)
    settled_dedup = _prep(settled_dedup)
    rows = []

    def _summ(grouping, value, sub):
        n = len(sub)
        won = int(sub["won_flag"].sum())
        roi = sub["profit_num"].mean() if sub["profit_num"].notna().any() else None
        avg_odds = sub["odds_num"].mean() if sub["odds_num"].notna().any() else None
        be = (1.0 / avg_odds) if avg_odds and avg_odds > 1 else None
        hit = (won / n) if n else None
        stakes = sub["stake_num"]
        money = sub["profit_money_num"]
        if money.notna().any() and stakes.notna().any() and stakes.fillna(0).sum() > 0:
            roi_sw = float(money.fillna(0).sum() / stakes.fillna(0).sum())
        else:
            roi_sw = None
        avg_clv = sub["clv_num"].mean() if sub["clv_num"].notna().any() else None
        note = "LOW SAMPLE" if n < SAMPLE_SIZE_WARNING else ""
        if note == "" and hit is not None and be is not None:
            if hit + 0.03 < be:
                note = "BELOW BREAK-EVEN"
            elif hit > be + 0.05:
                note = "BEATING BREAK-EVEN"
        rows.append({
            "grouping": grouping, "value": value, "bets": n, "won": won, "lost": n - won,
            "hit_rate": f"{round(won / n * 100)}%" if n else "",
            "avg_prob": round(sub["prob_num"].mean(), 3) if sub["prob_num"].notna().any() else "",
            "avg_odds": round(avg_odds, 2) if avg_odds is not None and avg_odds == avg_odds else "",
            "break_even": f"{round(be * 100)}%" if be is not None else "",
            "roi_per_bet": round(roi, 3) if roi is not None else "",
            "roi_stake_wtd": round(roi_sw, 3) if roi_sw is not None else "",
            "avg_clv": round(avg_clv, 4) if avg_clv is not None and avg_clv == avg_clv else "",
            "note": note,
        })

    # De-duped headline figures
    _summ("OVERALL", "all bets (de-duped)", settled_dedup)
    recent_cut = (now() - timedelta(days=30)).strftime("%Y-%m-%d")
    recent = settled_dedup[settled_dedup["utcDate"].astype(str).str[:10] >= recent_cut]
    if not recent.empty:
        _summ("OVERALL", "last 30 days (de-duped)", recent)

    # By source keeps every logged row (you want tab-level truth)
    for src, sub in settled.groupby("source"):
        _summ("by source tab", str(src), sub)
    for bet, sub in settled_dedup.groupby("bet"):
        _summ("by bet type", str(bet), sub)
    for lg, sub in settled_dedup.groupby("league"):
        if str(lg).strip():
            _summ("by league", str(lg), sub)
    for osrc, sub in settled_dedup.groupby("odds_source"):
        label = str(osrc).strip() or "unknown"
        _summ("by odds source", label, sub)

    bracket_order = [
        "1.01 – 1.49 (heavy fav)",
        "1.50 – 1.79 (fav)",
        "1.80 – 2.19 (near even)",
        "2.20 – 2.79 (mild dog)",
        "2.80 – 3.49 (dog)",
        "3.50 – 4.99 (longshot)",
        "5.00+ (very long)",
        "unknown / missing",
    ]
    present = settled_dedup["odds_bracket"].value_counts()
    for br in bracket_order:
        if br in present.index:
            _summ("by odds bracket", br, settled_dedup[settled_dedup["odds_bracket"] == br])

    # Expected total corners (from All_Games_Best) vs actual
    rows.extend(_corners_accuracy_rows(settled))

    df = pd.DataFrame(rows)
    order = {
        "OVERALL": 0,
        "expected corners": 0.5,
        "by bet type": 1,
        "by source tab": 2,
        "by league": 3,
        "by odds source": 4,
        "by odds bracket": 5,
    }

    def _sort_key(row):
        g = row["grouping"]
        base = order.get(g, 9)
        if g == "by odds bracket":
            try:
                return (base, bracket_order.index(row["value"]))
            except ValueError:
                return (base, 99)
        if g == "OVERALL":
            # all bets first, then last 30 days
            return (base, 0 if "all bets" in str(row["value"]) else 1)
        if g == "expected corners":
            # "all matches" first, then leagues by sample size
            return (base, 0 if str(row["value"]) == "all matches" else 1, -int(row["bets"] or 0))
        return (base, -int(row["bets"]))

    df["_s"] = df.apply(_sort_key, axis=1)
    return df.sort_values("_s").drop(columns=["_s"]).reset_index(drop=True)



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
    try:
        fx_df = attach_corner_avgs(fx_df)
        log(f"corner avgs attached to {len(fx_df)} fixture(s)")
    except Exception as e:
        log(f"corner avgs skipped: {e}")
    write_df(TAB_FIXTURES, sort_by_date(fx_df))
    write_df(TAB_ACCESS, pd.DataFrame(access_rows))

    top10_specs = [("p_over_1_5", "OVER 1.5", "Top10_Over_1_5"),
                   ("p_over_2_5", "OVER 2.5", "Top10_Over_2_5")]
    visible_tabs = [TAB_ALL_GAMES, TAB_BEST_BETS, "Top10_Over_1_5", "Top10_Over_2_5",
                    TAB_CORNERS_O95, TAB_COMBOS,
                    TAB_SAFE, TAB_BAL, TAB_SHORTLIST, TAB_DASHBOARD, TAB_ACCURACY]

    history = read_df(TAB_HISTORY)
    history = settle_history(history, rs_df)
    history = archive_old_pending(history)
    bankroll = apply_settled_to_bankroll(history, bankroll)

    def _finish(reason):
        write_df(TAB_HISTORY, history if history is not None and not history.empty else pd.DataFrame(),
                 corners_view=False)
        write_df(TAB_ACCURACY, build_accuracy(history))
        set_visible_tabs(visible_tabs)
        log(f"=== DONE ({reason}) ===")

    if rs_df.empty:
        for t in (TAB_TEAM_FORM, TAB_PICKS, TAB_SAFE, TAB_BAL, TAB_BEST_BETS, TAB_ALL_GAMES, TAB_CORNERS_O95, TAB_COMBOS, TAB_SHORTLIST, TAB_DASHBOARD):
            write_df(t, pd.DataFrame())
        for _, _, t in top10_specs:
            write_df(t, pd.DataFrame())
        _finish("no results")
        return

    league_base = compute_league_baselines(rs_df)
    team_idx = compute_team_indices(rs_df, league_base)
    write_df(TAB_TEAM_FORM, team_idx.reset_index())

    if fx_df.empty:
        for t in (TAB_PICKS, TAB_SAFE, TAB_BAL, TAB_BEST_BETS, TAB_ALL_GAMES, TAB_CORNERS_O95, TAB_COMBOS, TAB_SHORTLIST, TAB_DASHBOARD):
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
    try:
        probs_df = attach_corner_avgs(probs_df)
    except Exception as e:
        log(f"corner avgs on probs skipped: {e}")

    odds_map = {}
    try:
        odds_map = fetch_market_odds(fx_df)
    except Exception as e:
        log(f"odds integration skipped: {e}")

    # Update closing_odds on still-PENDING history rows while prices are live
    history = refresh_closing_odds(history, odds_map)

    for i, row in probs_df.iterrows():
        mkt = market_probs_for_fixture(odds_map.get((row["league"], row["home"], row["away"])), row["home"], row["away"])
        for col in ("p_home", "p_draw", "p_away"):
            probs_df.at[i, col] = blend(float(row[col]), mkt.get(col))
        probs_df.at[i, "p_1x"] = probs_df.at[i, "p_home"] + probs_df.at[i, "p_draw"]
        probs_df.at[i, "p_x2"] = probs_df.at[i, "p_draw"] + probs_df.at[i, "p_away"]
        probs_df.at[i, "p_12"] = probs_df.at[i, "p_home"] + probs_df.at[i, "p_away"]

    # ---------- All_Games_Best is the source of truth for expected total corners ----------
    # Built first so its expected corners are stamped onto every other tab and the history.
    all_games_raw = build_all_games_best(probs_df)
    set_corner_lookup(all_games_raw)
    log(f"expected total corners available for {len(_ALL_GAMES_CORNERS)} of "
        f"{len(all_games_raw) if all_games_raw is not None else 0} game(s)")

    # ---------- Top10 tabs ----------
    top10_frames = {}
    for col, label, tab in top10_specs:
        soft = top_n_for_market(probs_df, col, label, TOP_N)
        soft = attach_odds(soft, odds_map, bankroll, apply_filter=False)
        final = ensure_at_least_one(soft, soft, size=3)
        write_df(tab, sort_by_date(final))
        top10_frames[tab] = final

    # ---------- Over 9.5 Corners ----------
    log("Over_9_5_Corners...")
    soft_corners = attach_odds(
        build_over_corners(probs_df, line=9.5, top_n=TOP_N * 2),
        odds_map, bankroll, apply_filter=False,
    )
    strict_corners = attach_odds(
        build_over_corners(probs_df, line=9.5, top_n=TOP_N),
        odds_map, bankroll, apply_filter=False,
    )
    corners_df = ensure_at_least_one(strict_corners, soft_corners, size=3)
    write_df(TAB_CORNERS_O95, sort_by_date(corners_df))
    top10_frames[TAB_CORNERS_O95] = corners_df

    baselines = build_league_market_baselines(probs_df)

    # ---------- Safe_Picks ----------
    log("Safe_Picks...")
    soft_safe = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 30), odds_map, bankroll, apply_filter=False)
    strict_safe = attach_odds(make_filtered_picks(probs_df, baselines, SAFE_RULES, 20), odds_map, bankroll)
    strict_safe = apply_value_filter(strict_safe)
    strict_safe = apply_over15_quality_filter(strict_safe)
    strict_safe = apply_performance_filter(strict_safe, history)
    strict_safe = apply_odds_bracket_filter(strict_safe, history)
    strict_safe = apply_league_weighting(strict_safe, history)
    safe_df = _dedupe_matches(ensure_at_least_one(strict_safe, soft_safe))
    write_df(TAB_SAFE, sort_by_date(safe_df))

    # ---------- Balanced_Picks ----------
    log("Balanced_Picks...")
    soft_bal = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 40), odds_map, bankroll, apply_filter=False)
    strict_bal = attach_odds(make_filtered_picks(probs_df, baselines, BAL_RULES, 30), odds_map, bankroll)
    strict_bal = apply_value_filter(strict_bal)
    strict_bal = apply_over15_quality_filter(strict_bal)
    strict_bal = apply_performance_filter(strict_bal, history)
    strict_bal = apply_odds_bracket_filter(strict_bal, history)
    strict_bal = apply_league_weighting(strict_bal, history)
    bal_df = _dedupe_matches(ensure_at_least_one(strict_bal, soft_bal))
    write_df(TAB_BAL, sort_by_date(bal_df))

    # ---------- Best_Bets ----------
    log("Best_Bets...")
    soft_best = attach_odds(build_best_bets(probs_df), odds_map, bankroll, apply_filter=False)
    strict_best = attach_odds(build_best_bets(probs_df), odds_map, bankroll)
    strict_best = apply_value_filter(strict_best)
    strict_best = apply_over15_quality_filter(strict_best)
    strict_best = apply_performance_filter(strict_best, history)
    strict_best = apply_odds_bracket_filter(strict_best, history)
    strict_best = apply_league_weighting(strict_best, history)
    best_df = _dedupe_matches(ensure_at_least_one(strict_best, soft_best))
    write_df(TAB_BEST_BETS, sort_by_date(best_df))

    # ---------- All_Games_Best (every fixture, most probable outcome, sorted by league) ----------
    log("All_Games_Best...")
    all_games_df = attach_odds(all_games_raw, odds_map, bankroll, apply_filter=False)
    write_df(TAB_ALL_GAMES, all_games_df)

    # ---------- Combo_Bets ----------
    log("Combo_Bets...")
    soft_combos = attach_odds(build_combo_bets(probs_df, top_k=40), odds_map, bankroll, apply_filter=False)
    strict_combos = attach_odds(build_combo_bets(probs_df, top_k=COMBO_TOP_N), odds_map, bankroll, apply_filter=False)
    combos_df = _dedupe_matches(ensure_at_least_one(strict_combos, soft_combos))
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
    for tab, frame in top10_frames.items():
        history = append_new_picks(history, frame, tab)
    if history is not None and not history.empty:
        for c in HISTORY_COLS:
            if c not in history.columns:
                history[c] = ""
        history = history[HISTORY_COLS]
    write_df(TAB_HISTORY, history, corners_view=False)
    write_df(TAB_ACCURACY, build_accuracy(history))

    set_visible_tabs(visible_tabs)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
