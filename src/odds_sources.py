"""
Multi-source bookmaker odds so pick tabs never stay blank.

Priority order when resolving a price for a bet:
  1. The Odds API  (live, needs ODDS_API_KEY)
  2. football-data.co.uk season CSVs  (Bet365 + market average, free, no key)
  3. Scrapling  (Paddy Power scrape, optional)
  4. Model-implied  (1 / model_probability)  — always available

football-data.co.uk columns used:
  B365H / B365D / B365A   Bet365 1X2
  B365>2.5 / B365<2.5     Bet365 totals
  AvgH / AvgD / AvgA      market average 1X2
  Avg>2.5 / Avg<2.5       market average totals
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# football-data.org league code → football-data.co.uk CSV code
FD_TO_CSV = {
    "PL": "E0",
    "ELC": "E1",
    "EL1": "E2",
    "EL2": "E3",
    "NL": "E3",   # closest free file; National League not always published
    "PD": "SP1",
    "SD": "SP2",
    "SA": "I1",
    "BL1": "D1",
    "FL1": "F1",
    "DED": "N1",
    "PPL": "P1",
    "SPL": "SC0",
    "BSA": "BR1",
}

# season folder on football-data.co.uk (update each August)
CURRENT_SEASON = "2526"  # 2025-26


def _normalize(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"\b(fc|cf|afc|ac|sc|club|united|city|town|hotspur|wanderers)\b", "", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _season_url(csv_code: str, season: str = CURRENT_SEASON) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{csv_code}.csv"


class FootballDataOdds:
    """Cache of Bet365 / average odds from football-data.co.uk CSVs."""

    def __init__(self):
        self._by_league: Dict[str, pd.DataFrame] = {}
        self._loaded: set = set()

    def load_league(self, league: str) -> None:
        csv_code = FD_TO_CSV.get(league)
        if not csv_code or csv_code in self._loaded:
            return
        self._loaded.add(csv_code)
        url = _season_url(csv_code)
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                return
            # Normalise team names for matching
            if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
                df = df.copy()
                df["_hn"] = df["HomeTeam"].map(_normalize)
                df["_an"] = df["AwayTeam"].map(_normalize)
                self._by_league[league] = df
                logger.info("football-data.co.uk loaded %s (%s rows) from %s", league, len(df), csv_code)
        except Exception as e:
            logger.warning("football-data.co.uk load failed for %s: %s", league, e)

    def load_for_fixtures(self, leagues: List[str]) -> None:
        for lg in set(leagues):
            self.load_league(lg)

    def lookup(
        self, league: str, home: str, away: str
    ) -> Optional[Dict[str, Any]]:
        """Return odds dict for a fixture if found in the CSV cache."""
        if league not in self._by_league:
            self.load_league(league)
        df = self._by_league.get(league)
        if df is None or df.empty:
            return None

        hn, an = _normalize(home), _normalize(away)
        # Exact normalised match first
        mask = (df["_hn"] == hn) & (df["_an"] == an)
        sub = df[mask]
        if sub.empty:
            # fuzzy contains
            mask = df["_hn"].str.contains(hn[:6], na=False) & df["_an"].str.contains(an[:6], na=False)
            sub = df[mask]
        if sub.empty:
            return None

        row = sub.iloc[-1]  # latest row for that pairing

        def _f(col: str) -> Optional[float]:
            if col not in row.index:
                return None
            try:
                v = float(row[col])
                return v if v > 1.01 else None
            except Exception:
                return None

        h2h = {}
        # Prefer Bet365, fall back to market average
        bh, bd, ba = _f("B365H"), _f("B365D"), _f("B365A")
        if not bh:
            bh, bd, ba = _f("AvgH"), _f("AvgD"), _f("AvgA")
        if bh:
            h2h[home] = bh
        if bd:
            h2h["Draw"] = bd
        if ba:
            h2h[away] = ba

        totals: Dict[float, Dict[str, float]] = {}
        over = _f("B365>2.5") or _f("Avg>2.5")
        under = _f("B365<2.5") or _f("Avg<2.5")
        if over or under:
            totals[2.5] = {}
            if over:
                totals[2.5]["Over"] = over
            if under:
                totals[2.5]["Under"] = under

        if not h2h and not totals:
            return None

        return {
            "h2h": h2h,
            "named_h2h": h2h,
            "named_totals": totals,
            "all_totals": totals,
            "named_btts": {},
            "btts": {},
            "book": "bet365" if _f("B365H") else "market-avg",
            "source": "football-data.co.uk",
            "event": None,
        }


# Singleton cache used by main
_FD_ODDS = FootballDataOdds()


def get_fd_odds_cache() -> FootballDataOdds:
    return _FD_ODDS


def model_implied_odds(prob: Any) -> Optional[float]:
    """Convert model probability to decimal odds. Always available fallback."""
    try:
        p = float(prob)
        if p <= 0.01 or p >= 0.99:
            return None
        return round(1.0 / p, 3)
    except Exception:
        return None


def merge_odds_entry(
    primary: Optional[Dict[str, Any]],
    secondary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Fill gaps in primary from secondary (does not overwrite existing prices)."""
    if not primary:
        return secondary
    if not secondary:
        return primary
    out = dict(primary)
    for key in ("named_h2h", "h2h", "named_btts", "btts"):
        base = dict(out.get(key) or {})
        for k, v in (secondary.get(key) or {}).items():
            if k not in base and v:
                base[k] = v
        out[key] = base
    for key in ("named_totals", "all_totals"):
        base = dict(out.get(key) or {})
        for line, sides in (secondary.get(key) or {}).items():
            if line not in base:
                base[line] = dict(sides)
            else:
                for s, p in sides.items():
                    if s not in base[line] and p:
                        base[line][s] = p
        out[key] = base
    if not out.get("book") and secondary.get("book"):
        out["book"] = secondary["book"]
    return out
