import os
import json
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

MODEL_TAB = "FOOTBALL_MODEL"
PICKS_TAB = "FOOTBALL_PICKS"
STATUS_TAB = "FOOTBALL_STATUS"


def _append_status(sh, msg: str):
    try:
        ws = sh.worksheet(STATUS_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=STATUS_TAB, rows=400, cols=10)
        ws.append_row(["timestamp_utc", "message"])
    ws.append_row([datetime.now(timezone.utc).isoformat(), msg])


def _upsert_df(sh, tab: str, df: pd.DataFrame, rows: int = 4000, cols: int = 60):
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
    min_prob_over_market: float  # require p_model - p_market >= this


PROFILES: Dict[str, Profile] = {
    "conservative": Profile(
        min_ev=0.04, min_odds=1.70, max_odds=3.50, max_bets=3,
        kelly_fraction=0.15, max_stake_pct=0.01, min_games_team=8,
        min_prob_over_market=0.03,
    ),
    "balanced": Profile(
        min_ev=0.025, min_odds=1.60, max_odds=4.50, max_bets=6,
        kelly_fraction=0.25, max_stake_pct=0.02, min_games_team=5,
        min_prob_over_market=0.02,
    ),
    "aggressive": Profile(
        min_ev=0.015, min_odds=1.50, max_odds=6.00, max_bets=10,
        kelly_fraction=0.40, max_stake_pct=0.03, min_games_team=3,
        min_prob_over_market=0.01,
    ),
}


# ---------------- Model helpers ----------------
def _poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)


def _goal_probs(lam: float, max_goals: int = 10) -> np.ndarray:
    p = np.array([_poisson_pmf(k, lam) for k in range(max_goals + 1)], dtype=float)
    s = p.sum()
    return p / s if s > 0 else p


def probs_from_lambdas(lam_home: float, lam_away: float, max_goals: int = 10):
    ph = _goal_probs(lam_home, max_goals)
    pa = _goal_probs(lam_away, max_goals)
    grid = np.outer(ph, pa)

    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()

    # BTTS
    p_no = grid[0, :].sum() + grid[:, 0].sum() - grid[0, 0]
    p_yes = 1 - p_no

    # Totals lines
    goals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    def p_over(line: float) -> float:
        # line is x.5 so over means >= ceil(x.5+0.5) => >= (x+1)
        thr = int(line + 0.5) + 1  # for 2.5 -> 3
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
        out[f"p_over_{line}"] = po
        out[f"p_under_{line}"] = 1 - po
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
    if s <= 0:
        return (None, None)
    return (p1 / s, p2 / s)


def normalize_3(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return (None, None, None)
    s = p1 + p2 + p3
    if s <= 0:
        return (None, None, None)
    return (p1 / s, p2 / s, p3 / s)


# ---------------- Team-strength Poisson fit (time-decay + L2 shrink) ----------------
@dataclass
class FitParams:
    teams: List[str]
    attack: Dict[str, float]
    defense: Dict[str, float]
    home_adv: float


def fit_team_strength_poisson(
    df_done: pd.DataFrame,
    xi: float = 0.0035,      # time decay
    l2: float = 1.0          # shrinkage
) -> FitParams:
    """
    df_done columns: utcDate, home, away, home_goals, away_goals (completed only)
    """
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

    # params: attack[n], defense[n], home_adv
    x0 = np.zeros(2 * n + 1, dtype=float)
    x0[-1] = 0.15

    def unpack(x):
        a = x[:n].copy()
        d = x[n:2*n].copy()
        ha = float(x[-1])
        # identifiability constraint: sum(attack)=0
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
            # Poisson log-likelihood
            ll += w[i] * (
                r.home_goals * np.log(lam_h) - lam_h +
                r.away_goals * np.log(lam_a) - lam_a
            )
        reg = l2 * (np.sum(a*a) + np.sum(d*d))
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


# ---------------- Betting math ----------------
def ev_decimal(p: float, odds: float) -> float:
    b = odds - 1.0
    q = 1.0 - p
    return p * b - q


def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return float(max(0.0, f))


# ---------------- Main ----------------
def main():
    risk = os.getenv("RISK_PROFILE", "balanced").strip().lower()
    profile = PROFILES.get(risk, PROFILES["balanced"])

    sheet_id = os.getenv("SHEET_ID")
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    bankroll = float(os.getenv("BANKROLL_EUR", "1000"))

    if not sheet_id:
        raise RuntimeError("Missing SHEET_ID")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # Read matches history
    matches = _read_tab(sh, MATCHES_TAB)
    if matches.empty:
        raise RuntimeError(f"{MATCHES_TAB} is empty")

    # Clean types
    for c in ["home_goals", "away_goals"]:
        if c in matches.columns:
            matches[c] = pd.to_numeric(matches[c], errors="coerce")

    matches["utcDate"] = pd.to_datetime(matches["utcDate"], utc=True, errors="coerce")
    matches = matches.dropna(subset=["utcDate", "home", "away"])

    # completed and upcoming
    done = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    upcoming = matches[matches["status"].isin(["SCHEDULED", "TIMED"])].copy()
    # If status strings differ, also treat future games as upcoming
    upcoming2 = matches[matches["utcDate"] > pd.Timestamp.now(tz="UTC")].copy()
    if len(upcoming2) > len(upcoming):
        upcoming = upcoming2

    # Fit model
    params = fit_team_strength_poisson(done, xi=0.0035, l2=1.0)

    # Team sample counts (confidence filter)
    team_games = pd.concat([
        done[["home"]].rename(columns={"home": "team"}),
        done[["away"]].rename(columns={"away": "team"})
    ], ignore_index=True)
    team_counts = team_games["team"].value_counts().to_dict()

    # Read odds if present
    odds = pd.DataFrame()
    try:
        odds = _read_tab(sh, ODDS_TAB)
    except Exception:
        odds = pd.DataFrame()

    # Convert odds cols to numeric if they exist
    odds_cols = [
        "odds_1x2_home", "odds_1x2_draw", "odds_1x2_away",
        "odds_btts_yes", "odds_btts_no",
        "odds_ou25_over", "odds_ou25_under",
    ]
    for c in odds_cols:
        if c in odds.columns:
            odds[c] = pd.to_numeric(odds[c], errors="coerce")

    # Join upcoming with odds on home/away (best effort)
    pred_base = upcoming[["utcDate", "home", "away", "competition", "status"]].copy() if "competition" in upcoming.columns else upcoming[["utcDate","home","away","status"]].copy()
    if not odds.empty and "home" in odds.columns and "away" in odds.columns:
        pred = pred_base.merge(odds, on=["home", "away"], how="left", suffixes=("", "_odds"))
    else:
        pred = pred_base.copy()

    # Build model rows
    model_rows = []
    for r in pred.itertuples(index=False):
        lam_h, lam_a = predict_lambdas(params, getattr(r, "home"), getattr(r, "away"))
        probs = probs_from_lambdas(lam_h, lam_a)

        hg = team_counts.get(getattr(r, "home"), 0)
        ag = team_counts.get(getattr(r, "away"), 0)

        row = {
            "utcDate": getattr(r, "utcDate"),
            "home": getattr(r, "home"),
            "away": getattr(r, "away"),
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "home_games_in_fit": hg,
            "away_games_in_fit": ag,
            **probs,
        }

        # add odds if available
        for c in odds_cols:
            if hasattr(r, c):
                row[c] = getattr(r, c)
        model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df["utcDate"] = pd.to_datetime(model_df["utcDate"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    _upsert_df(sh, MODEL_TAB, model_df, rows=4000, cols=80)
    _append_status(sh, f"FOOTBALL_MODEL written. Upcoming games: {len(model_df)}. Risk={risk}")

    # ---------------- Picks (EV + filters) ----------------
    picks = []

    def add_pick(market: str, side: str, p: float, odds_val: float, p_mkt: Optional[float], row):
        if odds_val is None or np.isnan(odds_val):
            return
        if odds_val < profile.min_odds or odds_val > profile.max_odds:
            return
        if p is None or np.isnan(p):
            return
        ev = ev_decimal(p, odds_val)
        if ev < profile.min_ev:
            return
        if p_mkt is not None and not np.isnan(p_mkt):
            if (p - p_mkt) < profile.min_prob_over_market:
                return

        k = kelly_fraction(p, odds_val) * profile.kelly_fraction
        stake = min(bankroll * k, bankroll * profile.max_stake_pct)

        picks.append({
            "utcDate": row["utcDate"],
            "home": row["home"],
            "away": row["away"],
            "market": market,
            "selection": side,
            "odds": float(odds_val),
            "p_model": float(p),
            "p_market_fair": (float(p_mkt) if p_mkt is not None and not np.isnan(p_mkt) else ""),
            "ev": float(ev),
            "kelly_used": float(k),
            "stake_eur": float(stake),
            "risk_profile": risk,
        })

    # Market fair probs from odds (if present)
    def fair_1x2(row):
        pH = implied_prob(row.get("odds_1x2_home"))
        pD = implied_prob(row.get("odds_1x2_draw"))
        pA = implied_prob(row.get("odds_1x2_away"))
        return normalize_3(pH, pD, pA)

    def fair_2way(o1, o2):
        p1 = implied_prob(o1)
        p2 = implied_prob(o2)
        return normalize_2(p1, p2)

    for _, row in model_df.iterrows():
        # confidence filter
        if int(row.get("home_games_in_fit", 0)) < profile.min_games_team or int(row.get("away_games_in_fit", 0)) < profile.min_games_team:
            continue

        # 1X2 picks
        pH, pD, pA = row["p_1x2_home"], row["p_1x2_draw"], row["p_1x2_away"]
        if "odds_1x2_home" in row and not pd.isna(row["odds_1x2_home"]):
            mH, mD, mA = fair_1x2(row)
            add_pick("1X2", "HOME", pH, row["odds_1x2_home"], mH, row)
            if "odds_1x2_draw" in row:
                add_pick("1X2", "DRAW", pD, row["odds_1x2_draw"], mD, row)
            if "odds_1x2_away" in row:
                add_pick("1X2", "AWAY", pA, row["odds_1x2_away"], mA, row)

        # BTTS picks
        if "odds_btts_yes" in row and not pd.isna(row["odds_btts_yes"]):
            mYes, mNo = fair_2way(row.get("odds_btts_yes"), row.get("odds_btts_no"))
            add_pick("BTTS", "YES", row["p_btts_yes"], row["odds_btts_yes"], mYes, row)
            if "odds_btts_no" in row and not pd.isna(row["odds_btts_no"]):
                add_pick("BTTS", "NO", row["p_btts_no"], row["odds_btts_no"], mNo, row)

        # O/U 2.5 (works if you have those odds)
        if "odds_ou25_over" in row and not pd.isna(row["odds_ou25_over"]):
            mOv, mUn = fair_2way(row.get("odds_ou25_over"), row.get("odds_ou25_under"))
            add_pick("O/U 2.5", "OVER", row["p_over_2.5"], row["odds_ou25_over"], mOv, row)
            if "odds_ou25_under" in row and not pd.isna(row["odds_ou25_under"]):
                add_pick("O/U 2.5", "UNDER", row["p_under_2.5"], row["odds_ou25_under"], mUn, row)

    picks_df = pd.DataFrame(picks)
    if not picks_df.empty:
        picks_df = picks_df.sort_values(["ev"], ascending=False).head(profile.max_bets)

    _upsert_df(sh, PICKS_TAB, picks_df, rows=1000, cols=30)
    _append_status(sh, f"FOOTBALL_PICKS written. Picks: {len(picks_df)}. Risk={risk}")

    print("Model + picks complete.")


if __name__ == "__main__":
    main()
