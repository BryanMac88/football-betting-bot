"""
FOOTBALL SHEETS BOT (robust)

Goals:
- Avoid manual errors (missing columns, wrong headers, mismatched odds naming)
- Support multiple O/U lines (0.5, 1.5, 2.5, 3.5, 4.5)
- Produce:
  - FOOTBALL_MODEL{suffix}
  - FOOTBALL_PICKS{suffix}
  - FOOTBALL_TOP_BETS{suffix}
  - FOOTBALL_BEST_PER_GAME{suffix}
  - FOOTBALL_JOIN_DIAG{suffix}
  - FOOTBALL_STATUS
- Optional cleanup to hide/delete legacy tabs that aren't owned by the bot

Required env vars:
- SHEET_ID
- GOOGLE_SERVICE_ACCOUNT_JSON   (service account JSON string)
Optional env vars:
- RISK_PROFILE: conservative|balanced|aggressive (default balanced)
- BANKROLL_EUR: float (default 1000)
- TAB_SUFFIX: e.g. "_BAL" (default "")
- CLEANUP_MODE: hide|delete|off (default hide)
- PROTECT_TABS: comma-separated tab names never touched by cleanup
- STRICT_SCHEMA: 1 to raise on missing required columns, else log and continue (default 0)
- MAX_BEST_PER_GAME: how many picks per match in BEST_PER_GAME (default 2)
"""

import os
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, factorial
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from scipy.optimize import minimize


# ---------------- Sheets ----------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MATCHES_TAB = "FOOTBALL_MATCHES"
ODDS_TAB = "FOOTBALL_ODDS"

MODEL_TAB_BASE = "FOOTBALL_MODEL"
PICKS_TAB_BASE = "FOOTBALL_PICKS"
TOP_TAB_BASE = "FOOTBALL_TOP_BETS"
BEST_GAME_TAB_BASE = "FOOTBALL_BEST_PER_GAME"
DIAG_TAB_BASE = "FOOTBALL_JOIN_DIAG"
STATUS_TAB = "FOOTBALL_STATUS"


# ---------------- Robust helpers ----------------
def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


def _to_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _append_status(sh, msg: str):
    try:
        ws = sh.worksheet(STATUS_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=STATUS_TAB, rows=800, cols=10)
        ws.append_row(["timestamp_utc", "message"])
    ws.append_row([datetime.now(timezone.utc).isoformat(), msg])


def _get_or_create_ws(sh, tab: str, rows: int = 4000, cols: int = 120):
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows=rows, cols=cols)


def _upsert_df(sh, tab: str, df: pd.DataFrame, rows: int = 4000, cols: int = 120):
    ws = _get_or_create_ws(sh, tab, rows=rows, cols=cols)

    if df is None or df.empty:
        ws.clear()
        ws.update([["no data"]])
        return

    # Convert to strings for Sheets, keep header
    values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
    ws.clear()
    ws.update(values)


def _read_tab(sh, tab: str) -> pd.DataFrame:
    ws = sh.worksheet(tab)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)


def _canonical_col(col: str) -> str:
    # Lowercase, trim, collapse spaces, convert punctuation to underscores
    c = str(col).strip().lower()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    new_cols = []
    used = set()
    for c in df.columns:
        cc = _canonical_col(c)
        # prevent duplicates after normalization
        if cc in used:
            i = 2
            while f"{cc}_{i}" in used:
                i += 1
            cc = f"{cc}_{i}"
        used.add(cc)
        new_cols.append(cc)
    df = df.copy()
    df.columns = new_cols
    return df


def _require_cols(
    df: pd.DataFrame,
    required: Iterable[str],
    context: str,
    sh=None,
    strict: bool = False
) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"[SCHEMA] Missing columns in {context}: {missing}"
        if sh is not None:
            _append_status(sh, msg)
        if strict:
            raise RuntimeError(msg)
        return False
    return True


def _ensure_odds_schema(df: pd.DataFrame, sh=None) -> pd.DataFrame:
    """
    Accepts lots of likely user headers and maps them to canonical columns the bot uses.
    This avoids manual errors with naming like 'OU15 Over', 'Over 1.5', etc.
    """
    if df is None or df.empty:
        return df

    df = _normalize_headers(df)

    # canonical required keys for matching
    # we match on home/away_norm anyway, but odds table needs home+away at minimum
    # We'll accept a few common variants.
    rename_map = {}

    # home/away variants
    for cand in ("home_team", "hometeam", "home"):
        if cand in df.columns and "home" not in df.columns:
            rename_map[cand] = "home"
            break
    for cand in ("away_team", "awayteam", "away"):
        if cand in df.columns and "away" not in df.columns:
            rename_map[cand] = "away"
            break

    # 1X2 variants
    variants = {
        "odds_1x2_home": ["odds_1x2_home", "odds_home", "home_odds", "h_odds", "1x2_home", "1x2_h", "odds_h"],
        "odds_1x2_draw": ["odds_1x2_draw", "odds_draw", "draw_odds", "d_odds", "1x2_draw", "1x2_d", "odds_d"],
        "odds_1x2_away": ["odds_1x2_away", "odds_away", "away_odds", "a_odds", "1x2_away", "1x2_a", "odds_a"],
        "odds_btts_yes": ["odds_btts_yes", "btts_yes", "btts_y", "btts_yes_odds", "odds_btts_y"],
        "odds_btts_no":  ["odds_btts_no", "btts_no", "btts_n", "btts_no_odds", "odds_btts_n"],
    }

    # O/U variants: we’ll accept a bunch and map them to odds_ouXX_over/under
    # For each line, build a list of possible header forms.
    ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    for line in ou_lines:
        key_over = f"odds_ou{int(line*10):02d}_over"
        key_under = f"odds_ou{int(line*10):02d}_under"
        # common names
        over_forms = [
            key_over,
            f"odds_ou_{line}_over",
            f"ou{line}_over",
            f"over_{line}",
            f"o_{line}",
            f"odds_over_{line}",
            f"odds_o{line}",
            f"ou_{int(line*10):02d}_over",
        ]
        under_forms = [
            key_under,
            f"odds_ou_{line}_under",
            f"ou{line}_under",
            f"under_{line}",
            f"u_{line}",
            f"odds_under_{line}",
            f"odds_u{line}",
            f"ou_{int(line*10):02d}_under",
        ]
        variants[key_over] = over_forms
        variants[key_under] = under_forms

    for canonical, cands in variants.items():
        if canonical in df.columns:
            continue
        for cand in cands:
            if cand in df.columns:
                rename_map[cand] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # log what we have
    if sh is not None:
        _append_status(sh, f"[ODDS] Columns after normalization: {sorted(df.columns)[:40]}{' ...' if len(df.columns)>40 else ''}")

    return df


def _cleanup_tabs(sh, keep: List[str], mode: str = "hide", protect: Optional[List[str]] = None):
    """
    Hide or delete tabs NOT in keep list.
    mode: "hide" (safe), "delete" (permanent), "off" (do nothing)
    protect: extra tabs that should NEVER be touched
    """
    mode = (mode or "hide").strip().lower()
    if mode in ("off", "none", "0", "false"):
        return

    keep_set = set(keep)
    protect_set = set(protect or [])

    # Guard: only touch likely legacy betting/view tabs
    def is_candidate(title: str) -> bool:
        t = title.strip().lower()
        if title.startswith("FOOTBALL_"):
            return True
        return t.startswith(("over", "under", "o/u", "ou", "btts", "1x2", "top", "picks", "model"))

    for ws in sh.worksheets():
        title = ws.title
        if title in keep_set or title in protect_set:
            continue
        if not is_candidate(title):
            continue

        if mode == "hide":
            try:
                ws.hide()
            except Exception:
                pass
        elif mode == "delete":
            sh.del_worksheet(ws)
        else:
            raise ValueError("CLEANUP_MODE must be 'hide', 'delete', or 'off'")


# ---------------- Team name normalization ----------------
_STOP_WORDS = [
    "fc", "cf", "sc", "afc", "cd", "ac", "sv", "fk", "sk", "nk",
    "club", "de", "la", "el", "the"
]

_ALIAS = {
    "manchester united": "man utd",
    "man united": "man utd",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "wolverhampton wanderers": "wolves",
    "internazionale": "inter",
    "borussia mgladbach": "borussia monchengladbach",
}

def norm_team(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p for p in s.split(" ") if p and p not in _STOP_WORDS]
    s = " ".join(parts)
    return _ALIAS.get(s, s)


# ---------------- Risk profiles ----------------
@dataclass(frozen=True)
class Profile:
    min_ev: float
    min_odds: float
    max_odds: float
    max_bets: int
    kelly_fraction: float
    max_stake_pct: float
    min_games_team: int
    min_prob_over_market: float


PROFILES: Dict[str, Profile] = {
    "conservative": Profile(0.04, 1.70, 3.50, 3, 0.15, 0.01, 8, 0.03),
    "balanced":     Profile(0.025, 1.60, 4.50, 6, 0.25, 0.02, 5, 0.02),
    "aggressive":   Profile(0.015, 1.50, 6.00, 10, 0.40, 0.03, 3, 0.01),
}

GRADE_THRESHOLDS = {
    "conservative": {
        "STRONG": {"score": 0.55, "edge": 0.10, "conf": 0.68, "pen": 0.06},
        "MEDIUM": {"score": 0.52, "edge": 0.08, "conf": 0.64, "pen": 0.08},
        "WATCH":  {"score": 0.49, "edge": 0.05, "conf": 0.60, "pen": 0.12},
    },
    "balanced": {
        "STRONG": {"score": 0.52, "edge": 0.08, "conf": 0.63, "pen": 0.07},
        "MEDIUM": {"score": 0.49, "edge": 0.05, "conf": 0.60, "pen": 0.09},
        "WATCH":  {"score": 0.46, "edge": 0.03, "conf": 0.55, "pen": 0.14},
    },
    "aggressive": {
        "STRONG": {"score": 0.50, "edge": 0.06, "conf": 0.58, "pen": 0.10},
        "MEDIUM": {"score": 0.48, "edge": 0.04, "conf": 0.55, "pen": 0.12},
        "WATCH":  {"score": 0.45, "edge": 0.02, "conf": 0.50, "pen": 0.16},
    },
}

def grade_pick(risk: str, score: float, edge: float, conf: float, pen: float) -> Tuple[str, str]:
    t = GRADE_THRESHOLDS.get(risk, GRADE_THRESHOLDS["balanced"])

    def ok(level: str) -> bool:
        req = t[level]
        return (score >= req["score"] and edge >= req["edge"] and conf >= req["conf"] and pen <= req["pen"])

    if ok("STRONG"):
        return "STRONG", "BET"
    if ok("MEDIUM"):
        return "MEDIUM", "SMALL"
    if ok("WATCH"):
        return "WATCH", "WATCH"
    return "AVOID", "SKIP"


def grade_rank(grade: str) -> int:
    order = {"STRONG": 0, "MEDIUM": 1, "WATCH": 2, "AVOID": 3}
    return order.get(str(grade).upper().strip(), 9)


# ---------------- Model helpers ----------------
def _poisson_pmf(k: int, lam: float) -> float:
    return (lam**k) * exp(-lam) / factorial(k)

def _goal_probs(lam: float, max_goals: int = 10) -> np.ndarray:
    p = np.array([_poisson_pmf(k, lam) for k in range(max_goals + 1)], dtype=float)
    s = p.sum()
    return p / s if s > 0 else p

def probs_from_lambdas(lam_home: float, lam_away: float, max_goals: int = 10) -> Dict[str, float]:
    ph = _goal_probs(lam_home, max_goals)
    pa = _goal_probs(lam_away, max_goals)
    grid = np.outer(ph, pa)

    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()

    p_no = grid[0, :].sum() + grid[:, 0].sum() - grid[0, 0]
    p_yes = 1 - p_no

    goals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))

    def p_over(line: float) -> float:
        thr = int(line + 0.5) + 1
        return float(grid[goals >= thr].sum())

    out = {
        "p_1x2_home": float(p_home),
        "p_1x2_draw": float(p_draw),
        "p_1x2_away": float(p_away),
        "p_btts_yes": float(p_yes),
        "p_btts_no": float(p_no),
    }
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        po = p_over(line)
        out[f"p_over_{line}"] = float(po)
        out[f"p_under_{line}"] = float(1 - po)
    return out

def implied_prob(odds: Optional[float]) -> Optional[float]:
    try:
        o = float(odds)
        if o <= 1.0:
            return None
        return 1.0 / o
    except Exception:
        return None

def normalize_2(p1, p2):
    if p1 is None or p2 is None:
        return (None, None)
    s = p1 + p2
    return (None, None) if s <= 0 else (p1 / s, p2 / s)

def normalize_3(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return (None, None, None)
    s = p1 + p2 + p3
    return (None, None, None) if s <= 0 else (p1 / s, p2 / s, p3 / s)


# ---------------- Team-strength Poisson fit ----------------
@dataclass
class FitParams:
    teams: List[str]
    attack: Dict[str, float]
    defense: Dict[str, float]
    home_adv: float

def fit_team_strength_poisson(df_done: pd.DataFrame, xi: float = 0.0035, l2: float = 1.0) -> FitParams:
    df = df_done.copy()
    df["utcDate"] = pd.to_datetime(df["utcDate"], utc=True, errors="coerce")
    df = df.dropna(subset=["utcDate", "home", "away", "home_goals", "away_goals"])
    if df.empty:
        raise RuntimeError("No completed matches to fit model")

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    teams = sorted(set(df["home"]).union(set(df["away"])))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    now = datetime.now(timezone.utc)
    age_days = (now - df["utcDate"]).dt.total_seconds() / 86400.0
    w = np.exp(-xi * age_days.values)

    x0 = np.zeros(2 * n + 1, dtype=float)
    x0[-1] = 0.15

    def unpack(x):
        a = x[:n].copy()
        d = x[n:2 * n].copy()
        ha = float(x[-1])
        a -= a.mean()
        return a, d, ha

    def nll(x):
        a, d, ha = unpack(x)
        ll = 0.0
        for i, r in enumerate(df.itertuples(index=False)):
            hi = idx[r.home]
            ai = idx[r.away]
            lam_h = np.exp(ha + a[hi] + d[ai])
            lam_a = np.exp(a[ai] + d[hi])
            ll += w[i] * (r.home_goals * np.log(lam_h) - lam_h + r.away_goals * np.log(lam_a) - lam_a)
        reg = l2 * (np.sum(a * a) + np.sum(d * d))
        return -(ll - reg)

    res = minimize(nll, x0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Poisson fit failed: {res.message}")

    a, d, ha = unpack(res.x)
    return FitParams(
        teams=teams,
        attack={t: float(a[idx[t]]) for t in teams},
        defense={t: float(d[idx[t]]) for t in teams},
        home_adv=float(ha),
    )

def predict_lambdas(params: FitParams, home: str, away: str) -> Tuple[float, float]:
    a_h = params.attack.get(home, 0.0)
    d_h = params.defense.get(home, 0.0)
    a_a = params.attack.get(away, 0.0)
    d_a = params.defense.get(away, 0.0)
    lam_home = float(np.exp(params.home_adv + a_h + d_a))
    lam_away = float(np.exp(a_a + d_h))
    return lam_home, lam_away


# ---------------- Betting + grading metrics ----------------
def ev_decimal(p: float, odds: float) -> float:
    b = odds - 1.0
    q = 1.0 - p
    return p * b - q

def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return float(max(0.0, f))

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def confidence_from_games(home_games: int, away_games: int) -> float:
    m = min(home_games, away_games)
    return clamp(m / 20.0, 0.0, 1.0)

def penalty_from_conf(conf: float) -> float:
    gap = max(0.0, 0.70 - conf)
    return clamp(gap * 0.5, 0.0, 0.20)

def score_from_ev(ev: float) -> float:
    return clamp(0.45 + 3.5 * float(ev), 0.0, 1.0)


# ---------------- Main ----------------
def main():
    strict = _to_bool(_env("STRICT_SCHEMA", "0"))

    risk = _env("RISK_PROFILE", "balanced").strip().lower()
    profile = PROFILES.get(risk, PROFILES["balanced"])

    suffix = _env("TAB_SUFFIX", "").strip()
    model_tab = f"{MODEL_TAB_BASE}{suffix}"
    picks_tab = f"{PICKS_TAB_BASE}{suffix}"
    top_tab = f"{TOP_TAB_BASE}{suffix}"
    best_game_tab = f"{BEST_GAME_TAB_BASE}{suffix}"
    diag_tab = f"{DIAG_TAB_BASE}{suffix}"

    sheet_id = _env("SHEET_ID", "").strip()
    sa_json = _env("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    bankroll_raw = _env("BANKROLL_EUR", "").strip()
    bankroll = float(bankroll_raw) if bankroll_raw else 1000.0

    max_best_per_game = int(float(_env("MAX_BEST_PER_GAME", "2")))

    if not sheet_id:
        raise RuntimeError("Missing SHEET_ID")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # ---------------- READ + NORMALIZE MATCHES ----------------
    matches_raw = _read_tab(sh, MATCHES_TAB)
    if matches_raw.empty:
        raise RuntimeError(f"{MATCHES_TAB} is empty")

    matches = _normalize_headers(matches_raw)

    # schema: minimal columns needed
    _require_cols(matches, ["utcdate", "home", "away"], MATCHES_TAB, sh=sh, strict=True)

    # optional goals columns can be absent
    if "home_goals" in matches.columns:
        matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    if "away_goals" in matches.columns:
        matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")

    matches["utcdate"] = pd.to_datetime(matches["utcdate"], utc=True, errors="coerce")
    matches = matches.dropna(subset=["utcdate", "home", "away"])

    done = matches.dropna(subset=[c for c in ["home_goals", "away_goals"] if c in matches.columns]).copy()
    # if goals columns missing entirely, done will be empty - handle explicitly
    if "home_goals" not in matches.columns or "away_goals" not in matches.columns:
        raise RuntimeError("FOOTBALL_MATCHES must include home_goals and away_goals for completed matches fit.")

    done = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    upcoming = matches[matches["utcdate"] > pd.Timestamp.now(tz="UTC")].copy()

    if done.empty:
        raise RuntimeError("No completed matches available to fit model")

    # fit model
    # rename to expected internal columns for the fit
    done_fit = done.rename(columns={"utcdate": "utcDate"})
    params = fit_team_strength_poisson(done_fit, xi=0.0035, l2=1.0)

    # team counts for confidence
    team_games = pd.concat(
        [
            done_fit[["home"]].rename(columns={"home": "team"}),
            done_fit[["away"]].rename(columns={"away": "team"}),
        ],
        ignore_index=True,
    )
    team_counts = team_games["team"].value_counts().to_dict()

    # ---------------- READ + NORMALIZE ODDS ----------------
    try:
        odds_raw = _read_tab(sh, ODDS_TAB)
    except Exception:
        odds_raw = pd.DataFrame()

    if odds_raw is None or odds_raw.empty:
        odds = pd.DataFrame()
        _append_status(sh, f"[ODDS] {ODDS_TAB} empty or missing. Picks will be limited.")
    else:
        odds = _ensure_odds_schema(odds_raw, sh=sh)
        odds = _normalize_headers(odds)

    # odds columns we support (after normalization)
    odds_cols = [
        "odds_1x2_home", "odds_1x2_draw", "odds_1x2_away",
        "odds_btts_yes", "odds_btts_no",
        "odds_ou05_over", "odds_ou05_under",
        "odds_ou15_over", "odds_ou15_under",
        "odds_ou25_over", "odds_ou25_under",
        "odds_ou35_over", "odds_ou35_under",
        "odds_ou45_over", "odds_ou45_under",
    ]

    if not odds.empty:
        # require at least home/away for join; not strict by default
        _require_cols(odds, ["home", "away"], ODDS_TAB, sh=sh, strict=strict)

        for c in odds_cols:
            if c in odds.columns:
                odds[c] = pd.to_numeric(odds[c], errors="coerce")

    # ---------------- JOIN UPCOMING + ODDS ----------------
    upcoming2 = upcoming.copy()
    upcoming2["home_norm"] = upcoming2["home"].map(norm_team)
    upcoming2["away_norm"] = upcoming2["away"].map(norm_team)

    pred = upcoming2.copy()
    if not odds.empty and "home" in odds.columns and "away" in odds.columns:
        odds2 = odds.copy()
        odds2["home_norm"] = odds2["home"].map(norm_team)
        odds2["away_norm"] = odds2["away"].map(norm_team)
        odds2 = odds2.drop_duplicates(subset=["home_norm", "away_norm"], keep="last")
        pred = upcoming2.merge(odds2, on=["home_norm", "away_norm"], how="left", suffixes=("", "_odds"))

    diag = pd.DataFrame([{
        "risk": risk,
        "upcoming_games": int(len(upcoming2)),
        "odds_rows_in_tab": int(len(odds)) if isinstance(odds, pd.DataFrame) else 0,
        "upcoming_with_1x2_odds": int(pred["odds_1x2_home"].notna().sum()) if "odds_1x2_home" in pred.columns else 0,
        "schema_strict": int(strict),
    }])
    _upsert_df(sh, diag_tab, diag, rows=50, cols=30)
    _append_status(sh, f"{diag_tab} written. upcoming={len(upcoming2)} with_odds={int(diag['upcoming_with_1x2_odds'][0])}")

    # ---------------- BUILD MODEL TAB ----------------
    model_rows = []
    for r in pred.itertuples(index=False):
        home = getattr(r, "home")
        away = getattr(r, "away")

        lam_h, lam_a = predict_lambdas(params, home, away)
        probs = probs_from_lambdas(lam_h, lam_a)

        hg = int(team_counts.get(home, 0))
        ag = int(team_counts.get(away, 0))

        row = {
            "utcDate": getattr(r, "utcdate"),
            "home": home,
            "away": away,
            "home_norm": getattr(r, "home_norm"),
            "away_norm": getattr(r, "away_norm"),
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "home_games_in_fit": hg,
            "away_games_in_fit": ag,
            **probs,
        }

        # pass-through optional fields if present
        for optional in ("competition", "status"):
            if hasattr(r, optional):
                row[optional] = getattr(r, optional)

        for c in odds_cols:
            if hasattr(r, c):
                row[c] = getattr(r, c)

        model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df["utcDate"] = pd.to_datetime(model_df["utcDate"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    _upsert_df(sh, model_tab, model_df, rows=4000, cols=160)
    _append_status(sh, f"{model_tab} written. Upcoming games: {len(model_df)}. Risk={risk}")

    # ---------------- PICKS ----------------
    picks: List[Dict] = []

    def fair_1x2(row):
        pH = implied_prob(row.get("odds_1x2_home"))
        pD = implied_prob(row.get("odds_1x2_draw"))
        pA = implied_prob(row.get("odds_1x2_away"))
        return normalize_3(pH, pD, pA)

    def fair_2way(o1, o2):
        p1 = implied_prob(o1)
        p2 = implied_prob(o2)
        return normalize_2(p1, p2)

    def add_pick(
        market: str,
        selection: str,
        p_model: float,
        odds_val: float,
        p_mkt_fair: Optional[float],
        row: pd.Series
    ):
        if odds_val is None or pd.isna(odds_val):
            return
        odds_val = float(odds_val)
        if odds_val < profile.min_odds or odds_val > profile.max_odds:
            return

        p_model = float(p_model)
        ev = ev_decimal(p_model, odds_val)
        if ev < profile.min_ev:
            return

        if p_mkt_fair is not None and not pd.isna(p_mkt_fair):
            edge = p_model - float(p_mkt_fair)
            if edge < profile.min_prob_over_market:
                return
        else:
            edge = ev  # fallback

        k = kelly_fraction(p_model, odds_val) * profile.kelly_fraction
        stake = min(bankroll * k, bankroll * profile.max_stake_pct)

        hg = int(row.get("home_games_in_fit", 0))
        ag = int(row.get("away_games_in_fit", 0))
        conf = confidence_from_games(hg, ag)
        pen = penalty_from_conf(conf)
        score = score_from_ev(ev)

        grade, action = grade_pick(risk, score, float(edge), conf, pen)

        home_xg = float(row.get("lambda_home", np.nan))
        away_xg = float(row.get("lambda_away", np.nan))
        xg_total = home_xg + away_xg
        xg_diff = home_xg - away_xg

        # readable "why"
        market_fair_str = ""
        if p_mkt_fair is not None and not pd.isna(p_mkt_fair):
            market_fair_str = f"{float(p_mkt_fair):.3f}"
        else:
            market_fair_str = ""

        why_text = (
            f"model_p={p_model:.3f}"
            + (f" vs mkt_p={market_fair_str}" if market_fair_str else "")
            + f" | edge={float(edge):.3f} | ev={ev:.3f}"
            + f" | xG={xg_total:.2f} (H {home_xg:.2f} / A {away_xg:.2f})"
            + f" | conf={conf:.2f} pen={pen:.2f} | score={score:.2f}"
        )

        picks.append({
            "utcDate": row.get("utcDate", ""),
            "home": row.get("home", ""),
            "away": row.get("away", ""),
            "competition": row.get("competition", ""),
            "market": market,
            "selection": selection,
            "prob": p_model,
            "p_market_fair": (float(p_mkt_fair) if p_mkt_fair is not None and not pd.isna(p_mkt_fair) else ""),
            "edge": float(edge),
            "ev": float(ev),
            "odds": odds_val,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "xg_total": float(xg_total),
            "xg_diff": float(xg_diff),
            "confidence": conf,
            "penalty": pen,
            "score": score,
            "kelly_used": float(k),
            "stake_eur": float(stake),
            "grade": grade,
            "action": action,
            "why": why_text,
            "risk_profile": risk,
        })

    if model_df.empty:
        _upsert_df(sh, picks_tab, pd.DataFrame(), rows=1000, cols=120)
        _upsert_df(sh, top_tab, pd.DataFrame(), rows=200, cols=120)
        _upsert_df(sh, best_game_tab, pd.DataFrame(), rows=500, cols=140)
        _append_status(sh, f"{picks_tab} empty (no upcoming games).")
        return

    # Generate picks across markets
    ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5]

    for _, row in model_df.iterrows():
        if int(row.get("home_games_in_fit", 0)) < profile.min_games_team or int(row.get("away_games_in_fit", 0)) < profile.min_games_team:
            continue

        # 1X2
        if not pd.isna(row.get("odds_1x2_home", np.nan)):
            mH, mD, mA = fair_1x2(row)
            add_pick("1X2", "HOME", row["p_1x2_home"], row.get("odds_1x2_home"), mH, row)
            add_pick("1X2", "DRAW", row["p_1x2_draw"], row.get("odds_1x2_draw"), mD, row)
            add_pick("1X2", "AWAY", row["p_1x2_away"], row.get("odds_1x2_away"), mA, row)

        # BTTS
        if not pd.isna(row.get("odds_btts_yes", np.nan)):
            mYes, mNo = fair_2way(row.get("odds_btts_yes"), row.get("odds_btts_no"))
            add_pick("BTTS", "YES", row["p_btts_yes"], row.get("odds_btts_yes"), mYes, row)
            add_pick("BTTS", "NO", row["p_btts_no"], row.get("odds_btts_no"), mNo, row)

        # O/U lines
        for line in ou_lines:
            over_col = f"odds_ou{int(line*10):02d}_over"
            under_col = f"odds_ou{int(line*10):02d}_under"
            if not pd.isna(row.get(over_col, np.nan)):
                mOv, mUn = fair_2way(row.get(over_col), row.get(under_col))
                add_pick(f"O/U {line}", "OVER", row[f"p_over_{line}"], row.get(over_col), mOv, row)
                add_pick(f"O/U {line}", "UNDER", row[f"p_under_{line}"], row.get(under_col), mUn, row)

    picks_df = pd.DataFrame(picks)

    # Sort + cap overall picks
    if not picks_df.empty:
        picks_df["grade_rank"] = picks_df["grade"].apply(grade_rank)
        picks_df = (
            picks_df.sort_values(["grade_rank", "score", "ev"], ascending=[True, False, False])
            .drop(columns=["grade_rank"])
            .head(profile.max_bets)
        )

    _upsert_df(sh, picks_tab, picks_df, rows=1000, cols=140)
    _append_status(sh, f"{picks_tab} written. Picks: {len(picks_df)}. Risk={risk}")

    # TOP BETS (BET + SMALL only, top 10)
    if picks_df is None or picks_df.empty:
        top_df = pd.DataFrame()
    else:
        top_df = picks_df[picks_df["action"].isin(["BET", "SMALL"])].copy()
        if not top_df.empty:
            top_df["grade_rank"] = top_df["grade"].apply(grade_rank)
            top_df = (
                top_df.sort_values(["grade_rank", "score", "ev"], ascending=[True, False, False])
                .drop(columns=["grade_rank"])
                .head(10)
            )

    _upsert_df(sh, top_tab, top_df, rows=200, cols=140)
    _append_status(sh, f"{top_tab} written. Top bets: {len(top_df)}. Risk={risk}")

    # BEST PICKS PER GAME (top N per match from full picks_df BEFORE top-10 cap would be ideal,
    # but we’re using capped picks_df for now to keep behavior consistent & simple)
    if picks_df is None or picks_df.empty:
        best_game_df = pd.DataFrame()
    else:
        temp = picks_df.copy()
        temp["grade_rank"] = temp["grade"].apply(grade_rank)
        temp = temp.sort_values(
            ["utcDate", "home", "away", "grade_rank", "score", "ev"],
            ascending=[True, True, True, True, False, False],
        )

        best_game_df = (
            temp.groupby(["utcDate", "home", "away"], as_index=False, sort=False)
            .head(max_best_per_game)
            .drop(columns=["grade_rank"])
            .copy()
        )

        # tighter explanation column for quick scan
        best_game_df["analysis"] = (
            best_game_df["market"].astype(str) + " " + best_game_df["selection"].astype(str)
            + " @ " + best_game_df["odds"].round(2).astype(str)
            + " | p=" + best_game_df["prob"].round(3).astype(str)
            + " edge=" + best_game_df["edge"].round(3).astype(str)
            + " ev=" + best_game_df["ev"].round(3).astype(str)
            + " | xG=" + best_game_df["xg_total"].round(2).astype(str)
        )

        # keep only the most useful columns up top (still output all columns, but this improves order)
        preferred = [
            "utcDate", "competition", "home", "away",
            "market", "selection", "odds", "prob", "p_market_fair", "edge", "ev",
            "xg_total", "xg_diff", "confidence", "penalty", "score",
            "grade", "action", "stake_eur", "analysis", "why", "risk_profile"
        ]
        # reorder columns if present
        cols = [c for c in preferred if c in best_game_df.columns] + [c for c in best_game_df.columns if c not in preferred]
        best_game_df = best_game_df[cols]

    _upsert_df(sh, best_game_tab, best_game_df, rows=500, cols=160)
    _append_status(sh, f"{best_game_tab} written. Rows: {len(best_game_df)}")

    # ---- CLEANUP legacy tabs ----
    keep_tabs = [
        MATCHES_TAB,
        ODDS_TAB,
        model_tab,
        picks_tab,
        top_tab,
        best_game_tab,
        diag_tab,
        STATUS_TAB,
    ]

    cleanup_mode = _env("CLEANUP_MODE", "hide").strip().lower()  # hide|delete|off
    protect_extra = [t.strip() for t in _env("PROTECT_TABS", "").split(",") if t.strip()]
    _cleanup_tabs(sh, keep_tabs, mode=cleanup_mode, protect=protect_extra)
    _append_status(sh, f"Cleanup complete. mode={cleanup_mode}. protected={len(protect_extra)}")


if __name__ == "__main__":
    main()
