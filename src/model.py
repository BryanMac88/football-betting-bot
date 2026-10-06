"""
Enhanced Poisson model with optional free xG blending.
Fully backward-compatible with the original goals-only path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import exp, factorial
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xg_data import blend_expected_goals, match_xg_to_fixture
    HAS_XG = True
except ImportError:
    HAS_XG = False


@dataclass
class ModelParams:
    rolling_window_matches: int = 12
    xg_weight: float = 0.65
    goals_weight: float = 0.35


def _poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * (lam ** k) / factorial(k)


def poisson_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        pi = _poisson_pmf(i, lam_home)
        for j in range(max_goals + 1):
            mat[i, j] = pi * _poisson_pmf(j, lam_away)
    s = mat.sum()
    return mat / s if s > 0 else mat


def build_team_form(results: pd.DataFrame, window: int) -> pd.DataFrame:
    res = results.dropna(subset=["Date", "home_team", "away_team"]).copy()
    res = res.sort_values("Date")
    teams = pd.unique(pd.concat([res["home_team"], res["away_team"]], ignore_index=True))
    rows = []
    for team in teams:
        home = res[res["home_team"] == team].copy()
        away = res[res["away_team"] == team].copy()

        def roll_mean(series):
            return series.tail(window).mean() if len(series) else np.nan

        rows.append({
            "team": team,
            "home_gf": roll_mean(home["home_goals"]),
            "home_ga": roll_mean(home["away_goals"]),
            "away_gf": roll_mean(away["away_goals"]),
            "away_ga": roll_mean(away["home_goals"]),
            "home_fouls": roll_mean(home.get("home_fouls", pd.Series(dtype=float))),
            "away_fouls": roll_mean(away.get("away_fouls", pd.Series(dtype=float))),
            "home_cards": roll_mean(
                home.get("home_yellows", pd.Series(dtype=float)).fillna(0)
                + 2 * home.get("home_reds", pd.Series(dtype=float)).fillna(0)
            ),
            "away_cards": roll_mean(
                away.get("away_yellows", pd.Series(dtype=float)).fillna(0)
                + 2 * away.get("away_reds", pd.Series(dtype=float)).fillna(0)
            ),
            "n_home": len(home.tail(window)),
            "n_away": len(away.tail(window)),
        })
    return pd.DataFrame(rows)


def expected_goals(
    form: pd.DataFrame,
    home: str,
    away: str,
    league_means: dict[str, float] | None = None,
    xg_table: Optional[pd.DataFrame] = None,
    params: Optional[ModelParams] = None,
) -> tuple[float, float]:
    params = params or ModelParams()
    league_means = league_means or {"home_g": 1.45, "away_g": 1.20}

    fh = form[form["team"] == home]
    fa = form[form["team"] == away]
    if fh.empty or fa.empty:
        return league_means["home_g"], league_means["away_g"]

    home_gf = float(fh["home_gf"].iloc[0]) if pd.notna(fh["home_gf"].iloc[0]) else league_means["home_g"]
    home_ga = float(fh["home_ga"].iloc[0]) if pd.notna(fh["home_ga"].iloc[0]) else league_means["away_g"]
    away_gf = float(fa["away_gf"].iloc[0]) if pd.notna(fa["away_gf"].iloc[0]) else league_means["away_g"]
    away_ga = float(fa["away_ga"].iloc[0]) if pd.notna(fa["away_ga"].iloc[0]) else league_means["home_g"]

    home_attack = home_gf / max(league_means["home_g"], 0.1)
    home_def = home_ga / max(league_means["away_g"], 0.1)
    away_attack = away_gf / max(league_means["away_g"], 0.1)
    away_def = away_ga / max(league_means["home_g"], 0.1)

    lam_h = home_attack * away_def * league_means["home_g"]
    lam_a = away_attack * home_def * league_means["away_g"]

    if HAS_XG and xg_table is not None and not xg_table.empty:
        h_xg, h_xga, a_xg, a_xga = match_xg_to_fixture(home, away, xg_table)
        if h_xg > 8:  # likely season total → rough per-match scale
            scale = 20.0
            h_xg, h_xga, a_xg, a_xga = h_xg / scale, h_xga / scale, a_xg / scale, a_xga / scale
        lam_h, lam_a = blend_expected_goals(
            goals_home_attack=home_attack,
            goals_away_defense=away_def,
            goals_away_attack=away_attack,
            goals_home_defense=home_def,
            xg_home_for=h_xg,
            xg_home_against=h_xga,
            xg_away_for=a_xg,
            xg_away_against=a_xga,
            xg_weight=params.xg_weight,
            goals_weight=params.goals_weight,
            league_home_mean=league_means["home_g"],
            league_away_mean=league_means["away_g"],
        )

    return max(0.2, min(4.0, float(lam_h))), max(0.2, min(4.0, float(lam_a)))


def probs_for_match(lam_home: float, lam_away: float, max_goals: int = 10) -> dict[str, float]:
    mat = poisson_matrix(lam_home, lam_away, max_goals=max_goals)
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    p_btts = float(mat[1:, 1:].sum())
    goals = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    p_over_0_5 = float(mat[goals >= 1].sum())
    p_over_1_5 = float(mat[goals >= 2].sum())
    p_over_2_5 = float(mat[goals >= 3].sum())
    p_over_3_5 = float(mat[goals >= 4].sum())
    p_btts_no = 1.0 - p_btts
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_1x": p_home + p_draw, "p_x2": p_draw + p_away, "p_12": p_home + p_away,
        "p_btts_yes": p_btts, "p_btts_no": p_btts_no,
        "p_over_0_5": p_over_0_5, "p_over_1_5": p_over_1_5,
        "p_over_2_5": p_over_2_5, "p_over_3_5": p_over_3_5,
        "p_under_1_5": 1.0 - p_over_1_5,
        "p_under_2_5": 1.0 - p_over_2_5,
        "p_under_3_5": 1.0 - p_over_3_5,
        "lam_home": lam_home, "lam_away": lam_away,
    }


def total_over_prob(mean_total: float, line: float, sd: float | None = None) -> float:
    sd = sd if sd and sd > 0 else max(1.0, math.sqrt(max(mean_total, 0.1)))
    z = (line + 0.5 - mean_total) / sd
    return float(0.5 * (1 - math.erf(z / math.sqrt(2))))


def build_predictions(
    fixtures: pd.DataFrame,
    form: pd.DataFrame,
    xg_tables: Optional[Dict[str, pd.DataFrame]] = None,
    params: Optional[ModelParams] = None,
) -> pd.DataFrame:
    params = params or ModelParams()
    rows = []
    for _, r in fixtures.iterrows():
        home = r.get("home_team") or r.get("home")
        away = r.get("away_team") or r.get("away")
        league = r.get("competition") or r.get("league") or ""
        xg_table = None
        if xg_tables and league in xg_tables:
            xg_table = xg_tables[league]

        lam_h, lam_a = expected_goals(form, home, away, xg_table=xg_table, params=params)
        p = probs_for_match(lam_h, lam_a)

        fh = form[form["team"] == home]
        fa = form[form["team"] == away]
        exp_fouls = exp_cards = np.nan
        if not fh.empty and not fa.empty:
            hf = fh["home_fouls"].iloc[0]
            af = fa["away_fouls"].iloc[0]
            if pd.notna(hf) and pd.notna(af):
                exp_fouls = float((hf + af) / 2.0)
            hc = fh["home_cards"].iloc[0]
            ac = fa["away_cards"].iloc[0]
            if pd.notna(hc) and pd.notna(ac):
                exp_cards = float((hc + ac) / 2.0)

        rows.append({
            **p,
            "competition": league,
            "commence_time": r.get("commence_time") or r.get("utcDate"),
            "home_team": home,
            "away_team": away,
            "home_xg": p["lam_home"],
            "away_xg": p["lam_away"],
            "exp_fouls": exp_fouls,
            "exp_cards": exp_cards,
        })
    out = pd.DataFrame(rows)
    for col in ["p_home", "p_draw", "p_away", "p_btts_yes", "p_over_2_5"]:
        if col in out.columns:
            out[f"fair_{col}"] = (1.0 / out[col]).replace([np.inf, -np.inf], np.nan)
    return out
