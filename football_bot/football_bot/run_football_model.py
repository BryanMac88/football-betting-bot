import os
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, factorial
from typing import Dict, Tuple, List, Optional

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
DIAG_TAB_BASE = "FOOTBALL_JOIN_DIAG"
STATUS_TAB = "FOOTBALL_STATUS"


def _append_status(sh, msg: str):
    try:
        ws = sh.worksheet(STATUS_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=STATUS_TAB, rows=800, cols=10)
        ws.append_row(["timestamp_utc", "message"])
    ws.append_row([datetime.now(timezone.utc).isoformat(), msg])


def _upsert_df(sh, tab: str, df: pd.DataFrame, rows: int = 4000, cols: int = 120):
    try:
        ws = sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=rows, cols=cols)

    if df is None or df.empty:
        ws.clear()
        ws.update([["no data"]])
        return

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


# ---------------- Team name normalization ----------------
_STOP_WORDS = [
    "fc", "cf", "sc", "afc", "cd", "ac", "sv", "fk", "sk", "nk",
    "club", "de", "la", "el", "the"
]

_ALIAS = {
    # Add any mappings you notice in your data here
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
    s = re.sub(r"[^\w\s]", " ", s)            # drop punctuation
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
    risk = os.getenv("RISK_PROFILE", "balanced").strip().lower()
    profile = PROFILES.get(risk, PROFILES["balanced"])

    suffix = os.getenv("TAB_SUFFIX", "").strip()
    model_tab = f"{MODEL_TAB_BASE}{suffix}"
    picks_tab = f"{PICKS_TAB_BASE}{suffix}"
    diag_tab = f"{DIAG_TAB_BASE}{suffix}"

    sheet_id = os.getenv("SHEET_ID")
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    bankroll_raw = os.getenv("BANKROLL_EUR", "").strip()
    bankroll = float(bankroll_raw) if bankroll_raw else 1000.0

    if not sheet_id:
        raise RuntimeError("Missing SHEET_ID")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    matches = _read_tab(sh, MATCHES_TAB)
    if matches.empty:
        raise RuntimeError(f"{MATCHES_TAB} is empty")

    for c in ["home_goals", "away_goals"]:
        if c in matches.columns:
            matches[c] = pd.to_numeric(matches[c], errors="coerce")

    matches["utcDate"] = pd.to_datetime(matches["utcDate"], utc=True, errors="coerce")
    matches = matches.dropna(subset=["utcDate", "home", "away"])

    done = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    upcoming = matches[matches["utcDate"] > pd.Timestamp.now(tz="UTC")].copy()
    if done.empty:
        raise RuntimeError("No completed matches available to fit model")

    params = fit_team_strength_poisson(done, xi=0.0035, l2=1.0)

    team_games = pd.concat(
        [
            done[["home"]].rename(columns={"home": "team"}),
            done[["away"]].rename(columns={"away": "team"}),
        ],
        ignore_index=True,
    )
    team_counts = team_games["team"].value_counts().to_dict()

    # Odds
    try:
        odds = _read_tab(sh, ODDS_TAB)
    except Exception:
        odds = pd.DataFrame()

    odds_cols = [
        "odds_1x2_home", "odds_1x2_draw", "odds_1x2_away",
        "odds_btts_yes", "odds_btts_no",
        "odds_ou25_over", "odds_ou25_under",
    ]
    for c in odds_cols:
        if c in odds.columns:
            odds[c] = pd.to_numeric(odds[c], errors="coerce")

    # Normalize team names in BOTH sets and merge on normalized keys
    upcoming = upcoming.copy()
    upcoming["home_norm"] = upcoming["home"].map(norm_team)
    upcoming["away_norm"] = upcoming["away"].map(norm_team)

    if not odds.empty and "home" in odds.columns and "away" in odds.columns:
        odds = odds.copy()
        odds["home_norm"] = odds["home"].map(norm_team)
        odds["away_norm"] = odds["away"].map(norm_team)

        # Keep latest odds per pairing if duplicates
        odds = odds.drop_duplicates(subset=["home_norm", "away_norm"], keep="last")

        pred = upcoming.merge(odds, on=["home_norm", "away_norm"], how="left", suffixes=("", "_odds"))
    else:
        pred = upcoming.copy()

    # Diagnostics: how many upcoming games have any odds?
    has_any_odds = False
    if all(c in pred.columns for c in ["odds_1x2_home", "odds_1x2_draw", "odds_1x2_away"]):
        has_any_odds = pred["odds_1x2_home"].notna().sum() > 0

    diag = pd.DataFrame([{
        "risk": risk,
        "upcoming_games": int(len(upcoming)),
        "odds_rows_in_tab": int(len(odds)) if isinstance(odds, pd.DataFrame) else 0,
        "upcoming_with_1x2_odds": int(pred["odds_1x2_home"].notna().sum()) if "odds_1x2_home" in pred.columns else 0,
        "note": "If upcoming_with_1x2_odds is 0, your odds feed is not matching team names or only covers other leagues.",
    }])
    _upsert_df(sh, diag_tab, diag, rows=50, cols=20)
    _append_status(sh, f"{diag_tab} written. upcoming={len(upcoming)} with_odds={diag.get('upcoming_with_1x2_odds', [0])[0]}")

    # Build model table for upcoming (even if odds missing)
    model_rows = []
    for r in pred.itertuples(index=False):
        home = getattr(r, "home")
        away = getattr(r, "away")

        lam_h, lam_a = predict_lambdas(params, home, away)
        probs = probs_from_lambdas(lam_h, lam_a)

        hg = int(team_counts.get(home, 0))
        ag = int(team_counts.get(away, 0))

        row = {
            "utcDate": getattr(r, "utcDate"),
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

        if hasattr(r, "competition"):
            row["competition"] = getattr(r, "competition")
        if hasattr(r, "status"):
            row["status"] = getattr(r, "status")

        for c in odds_cols:
            if hasattr(r, c):
                row[c] = getattr(r, c)

        model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df["utcDate"] = pd.to_datetime(model_df["utcDate"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    _upsert_df(sh, model_tab, model_df, rows=4000, cols=120)
    _append_status(sh, f"{model_tab} written. Upcoming games: {len(model_df)}. Risk={risk}")

    # ---------------- Picks (require odds to exist) ----------------
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

    def add_pick(market: str, selection: str, p_model: float, odds_val: float, p_mkt_fair: Optional[float], row: pd.Series):
        if odds_val is None or pd.isna(odds_val):
            return
        odds_val = float(odds_val)
        if odds_val < profile.min_odds or odds_val > profile.max_odds:
            return

        p_model = float(p_model)
        ev = ev_decimal(p_model, odds_val)
        if ev < profile.min_ev:
            return

        edge = None
        if p_mkt_fair is not None and not pd.isna(p_mkt_fair):
            edge = p_model - float(p_mkt_fair)
            if edge < profile.min_prob_over_market:
                return
        else:
            edge = ev

        k = kelly_fraction(p_model, odds_val) * profile.kelly_fraction
        stake = min(bankroll * k, bankroll * profile.max_stake_pct)

        hg = int(row.get("home_games_in_fit", 0))
        ag = int(row.get("away_games_in_fit", 0))
        conf = confidence_from_games(hg, ag)
        pen = penalty_from_conf(conf)
        score = score_from_ev(ev)

        grade, action = grade_pick(risk, score, float(edge), conf, pen)

        picks.append({
            "utcDate": row.get("utcDate", ""),
            "home": row.get("home", ""),
            "away": row.get("away", ""),
            "competition": row.get("competition", ""),
            "market": market,
            "selection": selection,
            "prob": p_model,
            "edge": float(edge),
            "home_xg": float(row.get("lambda_home", np.nan)),
            "away_xg": float(row.get("lambda_away", np.nan)),
            "confidence": conf,
            "penalty": pen,
            "score": score,
            "odds": odds_val,
            "p_market_fair": (float(p_mkt_fair) if p_mkt_fair is not None and not pd.isna(p_mkt_fair) else ""),
            "ev": float(ev),
            "kelly_used": float(k),
            "stake_eur": float(stake),
            "grade": grade,
            "action": action,
            "why": f"score={score:.3f} | edge={float(edge):.3f} | conf={conf:.3f} | pen={pen:.3f} | ev={ev:.3f}",
            "risk_profile": risk,
        })

    if model_df.empty:
        _upsert_df(sh, picks_tab, pd.DataFrame(), rows=1000, cols=60)
        _append_status(sh, f"{picks_tab} empty (no upcoming games).")
        return

    for _, row in model_df.iterrows():
        # confidence filter
        if int(row.get("home_games_in_fit", 0)) < profile.min_games_team or int(row.get("away_games_in_fit", 0)) < profile.min_games_team:
            continue

        # 1X2 (only if odds exist)
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

        # O/U 2.5
        if not pd.isna(row.get("odds_ou25_over", np.nan)):
            mOv, mUn = fair_2way(row.get("odds_ou25_over"), row.get("odds_ou25_under"))
            add_pick("O/U 2.5", "OVER", row["p_over_2.5"], row.get("odds_ou25_over"), mOv, row)
            add_pick("O/U 2.5", "UNDER", row["p_under_2.5"], row.get("odds_ou25_under"), mUn, row)

    picks_df = pd.DataFrame(picks)
    if not picks_df.empty:
        picks_df = picks_df.sort_values(["score", "ev"], ascending=False).head(profile.max_bets)

    _upsert_df(sh, picks_tab, picks_df, rows=1000, cols=80)
    _append_status(sh, f"{picks_tab} written. Picks: {len(picks_df)}. Risk={risk}")


if __name__ == "__main__":
    main()
