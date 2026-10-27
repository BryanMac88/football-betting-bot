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
  HC / AC                 home / away corners (for averages)
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



    # ---------- Corner averages (HC / AC) ----------
    def _team_corner_table(self, league: str) -> Dict[str, Dict[str, float]]:
        """Per normalised team name: home/away corners for & against."""
        if league not in self._by_league:
            self.load_league(league)
        df = self._by_league.get(league)
        if df is None or df.empty or "HC" not in df.columns or "AC" not in df.columns:
            return {}

        rows: Dict[str, Dict[str, list]] = {}
        for _, r in df.iterrows():
            try:
                hc = float(r["HC"])
                ac = float(r["AC"])
            except Exception:
                continue
            if hc != hc or ac != ac:  # NaN
                continue
            hn = str(r.get("_hn") or "")
            an = str(r.get("_an") or "")
            if hn:
                rows.setdefault(hn, {"hf": [], "ha": [], "af": [], "aa": []})
                rows[hn]["hf"].append(hc)  # corners for at home
                rows[hn]["ha"].append(ac)  # corners against at home
            if an:
                rows.setdefault(an, {"hf": [], "ha": [], "af": [], "aa": []})
                rows[an]["af"].append(ac)  # corners for away
                rows[an]["aa"].append(hc)  # corners against away

        out: Dict[str, Dict[str, float]] = {}
        for team, buckets in rows.items():
            def _m(vals):
                return sum(vals) / len(vals) if vals else float("nan")
            out[team] = {
                "home_for": _m(buckets["hf"]),
                "home_against": _m(buckets["ha"]),
                "away_for": _m(buckets["af"]),
                "away_against": _m(buckets["aa"]),
                "n_home": float(len(buckets["hf"])),
                "n_away": float(len(buckets["af"])),
            }
        return out

    def _find_team_key(self, table: Dict[str, Dict[str, float]], name: str) -> Optional[str]:
        key = _normalize(name)
        if not key:
            return None
        if key in table:
            return key
        # fuzzy: startswith / contains
        for k in table:
            if k.startswith(key[:6]) or key.startswith(k[:6]) or key in k or k in key:
                return k
        return None

    def league_avg_corners(self, league: str) -> float:
        """Season average total corners per match in this league."""
        if league not in self._by_league:
            self.load_league(league)
        df = self._by_league.get(league)
        if df is None or df.empty or "HC" not in df.columns or "AC" not in df.columns:
            return 10.0  # global rough default
        totals = pd.to_numeric(df["HC"], errors="coerce") + pd.to_numeric(df["AC"], errors="coerce")
        m = totals.mean()
        return float(m) if m == m else 10.0

    def expected_corners(self, league: str, home: str, away: str) -> Dict[str, float]:
        """Expected corners for a fixture from season HC/AC averages.

        Returns keys:
          home_corners_avg  — home team corners-for when playing at home
          away_corners_avg  — away team corners-for when playing away
          exp_home_corners  — blended expected corners for the home side
          exp_away_corners  — blended expected corners for the away side
          exp_total_corners — expected match total
          league_avg_corners — league season average total
        """
        table = self._team_corner_table(league)
        league_avg = self.league_avg_corners(league)
        half = league_avg / 2.0

        hk = self._find_team_key(table, home)
        ak = self._find_team_key(table, away)
        ht = table.get(hk) if hk else None
        at = table.get(ak) if ak else None

        home_for = ht["home_for"] if ht and ht["home_for"] == ht["home_for"] else half
        home_ag = ht["home_against"] if ht and ht["home_against"] == ht["home_against"] else half
        away_for = at["away_for"] if at and at["away_for"] == at["away_for"] else half
        away_ag = at["away_against"] if at and at["away_against"] == at["away_against"] else half

        # Blend attack (for) with opponent defence (against)
        exp_home = (home_for + away_ag) / 2.0
        exp_away = (away_for + home_ag) / 2.0
        exp_total = exp_home + exp_away

        return {
            "home_corners_avg": round(float(home_for), 2),
            "away_corners_avg": round(float(away_for), 2),
            "exp_home_corners": round(float(exp_home), 2),
            "exp_away_corners": round(float(exp_away), 2),
            "exp_total_corners": round(float(exp_total), 2),
            "league_avg_corners": round(float(league_avg), 2),
        }


# Singleton cache used by main
_FD_ODDS = FootballDataOdds()


def get_fd_odds_cache() -> FootballDataOdds:
    return _FD_ODDS


def attach_corner_avgs(df: pd.DataFrame) -> pd.DataFrame:
    """Add corner average columns to a fixtures / probs DataFrame."""
    if df is None or df.empty:
        return df
    cache = get_fd_odds_cache()
    leagues = list(df["league"].unique()) if "league" in df.columns else []
    cache.load_for_fixtures(leagues)

    cols = [
        "home_corners_avg", "away_corners_avg",
        "exp_home_corners", "exp_away_corners",
        "exp_total_corners", "league_avg_corners",
    ]
    rows = []
    for _, r in df.iterrows():
        try:
            stats = cache.expected_corners(
                str(r.get("league", "")),
                str(r.get("home", "")),
                str(r.get("away", "")),
            )
        except Exception:
            stats = {c: "" for c in cols}
        rows.append(stats)
    extra = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for c in cols:
        out[c] = extra[c]
    return out


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
