"""
Simple performance filter using historical results.
Prefers bet types that have been winning recently.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Optional


def calculate_bet_type_performance(history: pd.DataFrame, min_bets: int = 8) -> Dict[str, float]:
    """
    Returns a score for each bet type based on recent performance.
    Higher score = better recent results.
    """
    if history is None or history.empty:
        return {}

    # We only care about settled bets
    if "result" not in history.columns:
        return {}

    df = history.copy()
    df = df[df["result"].isin(["Win", "Loss", "Won", "Lost", True, False, "W", "L"])]

    if df.empty:
        return {}

    # Normalise result column
    df["won"] = df["result"].astype(str).str.lower().isin(["win", "won", "w", "true", "1"])

    # Group by bet type
    if "bet" not in df.columns:
        return {}

    stats = df.groupby("bet").agg(
        total_bets=("won", "count"),
        wins=("won", "sum")
    ).reset_index()

    stats = stats[stats["total_bets"] >= min_bets]

    if stats.empty:
        return {}

    stats["win_rate"] = stats["wins"] / stats["total_bets"]

    # Simple score: win rate (can be improved later with average odds)
    performance = dict(zip(stats["bet"], stats["win_rate"]))
    return performance


def apply_performance_filter(
    picks: pd.DataFrame,
    history: pd.DataFrame,
    min_win_rate: float = 0.52,
    min_bets: int = 8
) -> pd.DataFrame:
    """
    Keep only picks from bet types that are performing well.
    If not enough history exists, keep all picks.
    """
    if picks is None or picks.empty:
        return picks

    performance = calculate_bet_type_performance(history, min_bets=min_bets)

    if not performance:
        # Not enough history yet → keep everything
        return picks

    def is_good_type(bet_name: str) -> bool:
        rate = performance.get(str(bet_name), None)
        if rate is None:
            return True  # unknown type → allow
        return rate >= min_win_rate

    filtered = picks[picks["bet"].apply(is_good_type)].copy()

    # Re-rank if rank column exists
    if "rank" in filtered.columns and not filtered.empty:
        filtered = filtered.sort_values("rank" if "score" not in filtered.columns else "score", ascending=True)
        filtered["rank"] = range(1, len(filtered) + 1)

    return filtered.reset_index(drop=True)
