"""
Performance filter + league performance weighting.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple


def calculate_bet_type_performance(history: pd.DataFrame, min_bets: int = 8) -> Dict[str, float]:
    if history is None or history.empty:
        return {}
    if "result" not in history.columns or "bet" not in history.columns:
        return {}

    df = history.copy()
    df = df[df["result"].astype(str).str.upper().isin(["WON", "LOST", "WIN", "LOSS", "W", "L"])]
    if df.empty:
        return {}

    df["won"] = df["result"].astype(str).str.upper().isin(["WON", "WIN", "W"])
    stats = df.groupby("bet").agg(total_bets=("won", "count"), wins=("won", "sum")).reset_index()
    stats = stats[stats["total_bets"] >= min_bets]
    if stats.empty:
        return {}
    stats["win_rate"] = stats["wins"] / stats["total_bets"]
    return dict(zip(stats["bet"], stats["win_rate"]))


def calculate_league_performance(history: pd.DataFrame, min_bets: int = 6) -> Dict[Tuple[str, str], float]:
    """Returns {(league, bet): win_rate}"""
    if history is None or history.empty:
        return {}
    if not all(c in history.columns for c in ["result", "bet", "league"]):
        return {}

    df = history.copy()
    df = df[df["result"].astype(str).str.upper().isin(["WON", "LOST", "WIN", "LOSS", "W", "L"])]
    if df.empty:
        return {}

    df["won"] = df["result"].astype(str).str.upper().isin(["WON", "WIN", "W"])
    stats = (
        df.groupby(["league", "bet"])
        .agg(total_bets=("won", "count"), wins=("won", "sum"))
        .reset_index()
    )
    stats = stats[stats["total_bets"] >= min_bets]
    if stats.empty:
        return {}
    stats["win_rate"] = stats["wins"] / stats["total_bets"]
    return {(str(r["league"]), str(r["bet"])): float(r["win_rate"]) for _, r in stats.iterrows()}


def apply_performance_filter(
    picks: pd.DataFrame,
    history: pd.DataFrame,
    min_win_rate: float = 0.52,
    min_bets: int = 8,
) -> pd.DataFrame:
    if picks is None or picks.empty:
        return picks

    performance = calculate_bet_type_performance(history, min_bets=min_bets)
    if not performance:
        return picks

    def is_good_type(bet_name: str) -> bool:
        rate = performance.get(str(bet_name), None)
        if rate is None:
            return True
        return rate >= min_win_rate

    filtered = picks[picks["bet"].apply(is_good_type)].copy()
    if not filtered.empty:
        if "score" in filtered.columns:
            filtered = filtered.sort_values("score", ascending=False)
        filtered["rank"] = range(1, len(filtered) + 1)
    return filtered.reset_index(drop=True)


def apply_league_weighting(picks: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Boost or penalise score based on historical league + bet type performance."""
    if picks is None or picks.empty or "score" not in picks.columns:
        return picks

    league_perf = calculate_league_performance(history)
    if not league_perf:
        return picks

    out = picks.copy()
    adjustments = []
    for _, r in out.iterrows():
        key = (str(r.get("league", "")), str(r.get("bet", "")))
        rate = league_perf.get(key)
        if rate is None:
            adjustments.append(1.0)
        elif rate >= 0.60:
            adjustments.append(1.12)      # strong league → boost
        elif rate >= 0.55:
            adjustments.append(1.05)
        elif rate < 0.45:
            adjustments.append(0.85)      # weak league → penalise
        else:
            adjustments.append(1.0)

    out["score"] = out["score"] * pd.Series(adjustments, index=out.index)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out
