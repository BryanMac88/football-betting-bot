"""
Performance filter using historical results from the Accuracy / Bet_History tab.
Automatically suppresses bet types that are currently losing.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict


def calculate_bet_type_performance(history: pd.DataFrame, min_bets: int = 8) -> Dict[str, float]:
    """
    Returns win-rate for each bet type that has enough settled results.
    Higher score = better recent results.
    """
    if history is None or history.empty:
        return {}

    if "result" not in history.columns or "bet" not in history.columns:
        return {}

    df = history.copy()

    # Accept the actual values written by the bot
    df = df[df["result"].astype(str).str.upper().isin(["WON", "LOST", "WIN", "LOSS", "W", "L"])]

    if df.empty:
        return {}

    df["won"] = df["result"].astype(str).str.upper().isin(["WON", "WIN", "W"])

    stats = (
        df.groupby("bet")
        .agg(total_bets=("won", "count"), wins=("won", "sum"))
        .reset_index()
    )

    stats = stats[stats["total_bets"] >= min_bets]

    if stats.empty:
        return {}

    stats["win_rate"] = stats["wins"] / stats["total_bets"]
    return dict(zip(stats["bet"], stats["win_rate"]))


def apply_performance_filter(
    picks: pd.DataFrame,
    history: pd.DataFrame,
    min_win_rate: float = 0.52,
    min_bets: int = 8,
) -> pd.DataFrame:
    """
    Keep only picks from bet types that are currently performing well.
    If there is not enough history yet, all picks are kept.
    """
    if picks is None or picks.empty:
        return picks

    performance = calculate_bet_type_performance(history, min_bets=min_bets)

    if not performance:
        # Not enough settled bets yet → keep everything
        return picks

    def is_good_type(bet_name: str) -> bool:
        rate = performance.get(str(bet_name), None)
        if rate is None:
            return True          # unknown type → allow
        return rate >= min_win_rate

    filtered = picks[picks["bet"].apply(is_good_type)].copy()

    # Re-rank cleanly
    if not filtered.empty:
        if "score" in filtered.columns:
            filtered = filtered.sort_values("score", ascending=False)
        elif "rank" in filtered.columns:
            filtered = filtered.sort_values("rank", ascending=True)
        filtered["rank"] = range(1, len(filtered) + 1)

    return filtered.reset_index(drop=True)
