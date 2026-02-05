from __future__ import annotations
import numpy as np
import pandas as pd
from math import exp, factorial
from dataclasses import dataclass

@dataclass
class ModelParams:
    rolling_window_matches: int = 12

def _poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * (lam ** k) / factorial(k)

def poisson_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    mat = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        pi = _poisson_pmf(i, lam_home)
        for j in range(max_goals+1):
            mat[i,j] = pi * _poisson_pmf(j, lam_away)
    # Normalize small floating error
    return mat / mat.sum()

def build_team_form(results: pd.DataFrame, window: int) -> pd.DataFrame:
    # Rolling averages by team, split home/away
    res = results.dropna(subset=["Date","home_team","away_team"]).copy()
    res = res.sort_values("Date")
    teams = pd.unique(pd.concat([res["home_team"], res["away_team"]], ignore_index=True))
    rows = []
    for team in teams:
        home = res[res["home_team"]==team].copy()
        away = res[res["away_team"]==team].copy()

        def roll_mean(series):
            return series.tail(window).mean() if len(series) else np.nan

        rows.append({
            "team": team,
            "home_gf": roll_mean(home["home_goals"]),
            "home_ga": roll_mean(home["away_goals"]),
            "away_gf": roll_mean(away["away_goals"]),
            "away_ga": roll_mean(away["home_goals"]),
            "home_fouls": roll_mean(home["home_fouls"]),
            "away_fouls": roll_mean(away["away_fouls"]),
            "home_cards": roll_mean(home["home_yellows"].fillna(0) + 2*home["home_reds"].fillna(0)),
            "away_cards": roll_mean(away["away_yellows"].fillna(0) + 2*away["away_reds"].fillna(0)),
        })
    form = pd.DataFrame(rows)
    return form

def expected_goals(form: pd.DataFrame, home: str, away: str, league_means: dict[str,float] | None = None) -> tuple[float,float]:
    # Simple blend of team attack/defense; fall back to league means if missing.
    league_means = league_means or {"home_g":1.45,"away_g":1.20}
    fh = form[form["team"]==home]
    fa = form[form["team"]==away]
    if fh.empty or fa.empty:
        return league_means["home_g"], league_means["away_g"]
    home_gf = float(fh["home_gf"].iloc[0]) if pd.notna(fh["home_gf"].iloc[0]) else league_means["home_g"]
    home_ga = float(fh["home_ga"].iloc[0]) if pd.notna(fh["home_ga"].iloc[0]) else league_means["away_g"]
    away_gf = float(fa["away_gf"].iloc[0]) if pd.notna(fa["away_gf"].iloc[0]) else league_means["away_g"]
    away_ga = float(fa["away_ga"].iloc[0]) if pd.notna(fa["away_ga"].iloc[0]) else league_means["home_g"]

    # Attack vs opponent defense blend
    lam_home = (home_gf + away_ga) / 2.0
    lam_away = (away_gf + home_ga) / 2.0
    # Keep sane bounds
    lam_home = float(np.clip(lam_home, 0.2, 3.5))
    lam_away = float(np.clip(lam_away, 0.2, 3.5))
    return lam_home, lam_away

def probs_for_match(lam_home: float, lam_away: float) -> dict[str,float]:
    mat = poisson_matrix(lam_home, lam_away, max_goals=10)
    # 1X2
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    # BTTS
    p_btts = float(mat[1:,1:].sum())
    # Totals
    goals = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    p_over_2_5 = float(mat[goals >= 3].sum())
    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts,
        "p_over_2_5": p_over_2_5,
        "lam_home": lam_home,
        "lam_away": lam_away,
    }

def total_over_prob(mean_total: float, line: float, sd: float | None = None) -> float:
    # Normal approximation for totals (fouls/cards). If sd missing, use sqrt(mean) heuristic.
    import math
    sd = sd if sd and sd>0 else max(1.0, math.sqrt(max(mean_total, 0.1)))
    # P(X > line) with continuity correction
    z = (line + 0.5 - mean_total) / sd
    # 1 - Phi(z)
    return float(0.5 * (1 - math.erf(z / math.sqrt(2))))

def build_predictions(fixtures: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in fixtures.iterrows():
        home=r["home_team"]; away=r["away_team"]
        lam_h, lam_a = expected_goals(form, home, away)
        p = probs_for_match(lam_h, lam_a)

        # Fouls/cards expected totals (if form has them)
        fh = form[form["team"]==home]
        fa = form[form["team"]==away]
        exp_fouls = np.nan
        exp_cards = np.nan
        if not fh.empty and not fa.empty:
            hf = fh["home_fouls"].iloc[0]; af = fa["away_fouls"].iloc[0]
            if pd.notna(hf) and pd.notna(af):
                exp_fouls = float((hf + af)/2.0)
            hc = fh["home_cards"].iloc[0]; ac = fa["away_cards"].iloc[0]
            if pd.notna(hc) and pd.notna(ac):
                exp_cards = float((hc + ac)/2.0)

        rows.append({
            **{k:v for k,v in p.items()},
            "competition": r.get("competition"),
            "commence_time": r.get("commence_time"),
            "home_team": home,
            "away_team": away,
            "exp_fouls": exp_fouls,
            "exp_cards": exp_cards,
        })
    out = pd.DataFrame(rows)
    # fair odds
    for col in ["p_home","p_draw","p_away","p_btts_yes","p_over_2_5"]:
        out[f"fair_{col}"] = (1.0 / out[col]).replace([np.inf, -np.inf], np.nan)
    return out
