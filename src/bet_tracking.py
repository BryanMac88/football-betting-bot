"""
Prediction tracking + combined-bet probability maths.
Only Over 1.5 combos are allowed.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple


def poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_grid(lh: float, la: float, max_goals: int = 10) -> List[List[float]]:
    ph = [poisson(lh, i) for i in range(max_goals + 1)]
    pa = [poisson(la, j) for j in range(max_goals + 1)]
    return [[ph[i] * pa[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]


COMBO_SPECS: List[Tuple[str, Callable[[int, int], bool]]] = [
    ("HOME WIN & OVER 1.5",          lambda i, j: i > j and (i + j) > 1),
    ("AWAY WIN & OVER 1.5",          lambda i, j: j > i and (i + j) > 1),
    ("DOUBLE CHANCE 1X & OVER 1.5",  lambda i, j: i >= j and (i + j) > 1),
    ("DOUBLE CHANCE X2 & OVER 1.5",  lambda i, j: j >= i and (i + j) > 1),
    ("OVER 1.5 & BTTS YES",          lambda i, j: (i + j) > 1 and i > 0 and j > 0),
]


def combo_probs(lh: float, la: float, max_goals: int = 10) -> Dict[str, float]:
    grid = score_grid(lh, la, max_goals)
    out: Dict[str, float] = {}
    for label, cond in COMBO_SPECS:
        total = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if cond(i, j):
                    total += grid[i][j]
        out[label] = total
    return out


def _single_leg_result(
    leg: str, hg: int, ag: int, total_corners: Optional[int] = None
) -> Optional[bool]:
    b = leg.strip().upper()
    tot = hg + ag

    # Corner markets — require total_corners from match stats
    if "CORNER" in b:
        if total_corners is None:
            return None
        try:
            # "OVER 9.5 CORNERS" / "CORNERS OVER 9.5" / "UNDER 9.5 CORNERS"
            parts = b.replace("CORNERS", " ").replace("CORNER", " ").split()
            line = None
            side = None
            for i, tok in enumerate(parts):
                if tok in ("OVER", "UNDER"):
                    side = tok
                    if i + 1 < len(parts):
                        line = float(parts[i + 1])
                    break
            if line is None or side is None:
                return None
            return total_corners > line if side == "OVER" else total_corners < line
        except (ValueError, IndexError):
            return None

    if b == "HOME WIN":
        return hg > ag
    if b == "AWAY WIN":
        return ag > hg
    if b == "DRAW":
        return hg == ag
    if b in ("DOUBLE CHANCE 1X", "1X"):
        return hg >= ag
    if b in ("DOUBLE CHANCE X2", "X2"):
        return ag >= hg
    if b in ("DOUBLE CHANCE 12", "12"):
        return hg != ag
    if b == "BTTS YES":
        return hg > 0 and ag > 0
    if b == "BTTS NO":
        return hg == 0 or ag == 0
    if b.startswith("OVER "):
        try:
            return tot > float(b.split()[1])
        except (IndexError, ValueError):
            return None
    if b.startswith("UNDER "):
        try:
            return tot < float(b.split()[1])
        except (IndexError, ValueError):
            return None
    return None


def evaluate_bet(
    bet: str, hg: int, ag: int, total_corners: Optional[int] = None
) -> Optional[bool]:
    legs = [l for l in str(bet).split("&")]
    results = [_single_leg_result(l, hg, ag, total_corners=total_corners) for l in legs]
    if any(r is None for r in results):
        return None
    return all(results)


def decimal_to_fractional(dec: float) -> str:
    try:
        dec = float(dec)
    except (TypeError, ValueError):
        return ""
    if dec <= 1:
        return ""
    profit = dec - 1.0
    best = None
    for denom in range(1, 51):
        num = round(profit * denom)
        if num <= 0:
            continue
        err = abs(profit - num / denom)
        if best is None or err < best[0]:
            best = (err, num, denom)
    if best is None:
        return ""
    _, num, denom = best
    return f"{num}/{denom}"


def implied_prob(dec: float) -> Optional[float]:
    try:
        dec = float(dec)
    except (TypeError, ValueError):
        return None
    if dec <= 1:
        return None
    return 1.0 / dec
