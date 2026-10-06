"""
Free advanced stats layer (xG / xGA / form).

Primary: soccerdata (FBref + Understat) – mature, free, cached.
Fallback: Scrapling adaptive scrapers for Understat / FBref.
All sources are free.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import soccerdata as sd
    HAS_SOCCERDATA = True
except ImportError:
    HAS_SOCCERDATA = False
    logger.warning("soccerdata not installed – xG features limited")

try:
    from scrapling.fetchers import Fetcher, StealthyFetcher
    HAS_SCRAPLING = True
except ImportError:
    HAS_SCRAPLING = False
    logger.warning("scrapling not installed – adaptive scrapers disabled")


def _normalize_team(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"\b(fc|cf|afc|ac|sc|club|calcio|cd|ud|rc|as|ss|ssc|united|city)\b", "", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def fetch_understat_team_xg(league_code: str, season: Optional[str] = None) -> pd.DataFrame:
    if not HAS_SOCCERDATA:
        return pd.DataFrame()
    mapping = {
        "PL": "ENG-Premier League",
        "PD": "ESP-La Liga",
        "SA": "ITA-Serie A",
        "BL1": "GER-Bundesliga",
        "FL1": "FRA-Ligue 1",
    }
    us_league = mapping.get(league_code)
    if not us_league:
        return pd.DataFrame()
    try:
        understat = sd.Understat(leagues=us_league, seasons=season or "2425")
        stats = understat.read_team_season_stats()
        if stats is None or stats.empty:
            return pd.DataFrame()
        df = stats.reset_index() if isinstance(stats.index, pd.MultiIndex) else stats.copy()
        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if "xg" in cl and "against" not in cl and "np" not in cl:
                rename[c] = "xg_for"
            elif "xga" in cl or ("xg" in cl and "against" in cl):
                rename[c] = "xg_against"
            elif "team" in cl:
                rename[c] = "team"
        df = df.rename(columns=rename)
        if "team" not in df.columns:
            return pd.DataFrame()
        keep = ["team"]
        for col in ["xg_for", "xg_against"]:
            if col in df.columns:
                keep.append(col)
        out = df[keep].copy()
        out["team_norm"] = out["team"].apply(_normalize_team)
        out["xg_diff"] = out.get("xg_for", 0) - out.get("xg_against", 0)
        return out
    except Exception as e:
        logger.warning(f"Understat fetch failed for {league_code}: {e}")
        return pd.DataFrame()


def fetch_fbref_team_stats(league_key: str, season: Optional[str] = None) -> pd.DataFrame:
    if not HAS_SOCCERDATA:
        return pd.DataFrame()
    try:
        fbref = sd.FBref(leagues=league_key, seasons=season or "2425")
        std = fbref.read_team_season_stats(stat_type="standard")
        if std is None or std.empty:
            return pd.DataFrame()
        df = std.reset_index() if isinstance(std.index, pd.MultiIndex) else std.copy()
        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("xg", "expected goals", "xg_for"):
                rename[c] = "xg_for"
            elif cl in ("xga", "expected goals against", "xg_against"):
                rename[c] = "xg_against"
            elif "team" in cl:
                rename[c] = "team"
        df = df.rename(columns=rename)
        if "team" not in df.columns:
            return pd.DataFrame()
        df["team_norm"] = df["team"].apply(_normalize_team)
        return df
    except Exception as e:
        logger.warning(f"FBref fetch failed for {league_key}: {e}")
        return pd.DataFrame()


def scrapling_understat_league(league_slug: str = "EPL") -> pd.DataFrame:
    if not HAS_SCRAPLING:
        return pd.DataFrame()
    url = f"https://understat.com/league/{league_slug}"
    try:
        page = StealthyFetcher.fetch(url, headless=True, network_idle=True)
        rows = page.css("table tbody tr", adaptive=True)
        data = []
        for row in rows:
            cells = row.css("td")
            if len(cells) < 8:
                continue
            team = cells[1].get_all_text(strip=True) if len(cells) > 1 else ""
            try:
                xg = _safe_float(cells[9].get_all_text(strip=True)) if len(cells) > 9 else 0.0
                xga = _safe_float(cells[10].get_all_text(strip=True)) if len(cells) > 10 else 0.0
            except Exception:
                xg = xga = 0.0
            if team:
                data.append({
                    "team": team,
                    "team_norm": _normalize_team(team),
                    "xg_for": xg,
                    "xg_against": xga,
                    "xg_diff": xg - xga,
                })
        return pd.DataFrame(data)
    except Exception as e:
        logger.warning(f"Scrapling Understat failed: {e}")
        return pd.DataFrame()


def build_xg_team_strength(league_code: str, fbref_key: Optional[str] = None) -> pd.DataFrame:
    df = fetch_understat_team_xg(league_code)
    source = "understat"
    if df.empty and HAS_SCRAPLING:
        slug_map = {
            "PL": "EPL", "PD": "La_liga", "SA": "Serie_A",
            "BL1": "Bundesliga", "FL1": "Ligue_1",
        }
        slug = slug_map.get(league_code)
        if slug:
            df = scrapling_understat_league(slug)
            source = "understat-scrapling"
    if (df.empty or "xg_for" not in df.columns) and fbref_key:
        fb = fetch_fbref_team_stats(fbref_key)
        if not fb.empty and "xg_for" in fb.columns:
            df = fb
            source = "fbref"
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["source"] = source
    if "xg_for" not in df.columns:
        df["xg_for"] = 0.0
    if "xg_against" not in df.columns:
        df["xg_against"] = 0.0
    df["xg_diff"] = df["xg_for"] - df["xg_against"]
    return df[["team", "team_norm", "xg_for", "xg_against", "xg_diff", "source"]].drop_duplicates("team_norm")


def match_xg_to_fixture(home: str, away: str, xg_table: pd.DataFrame) -> Tuple[float, float, float, float]:
    if xg_table is None or xg_table.empty:
        return 1.4, 1.2, 1.2, 1.4
    nh, na = _normalize_team(home), _normalize_team(away)
    hrow = xg_table[xg_table["team_norm"] == nh]
    arow = xg_table[xg_table["team_norm"] == na]
    mean_xg = float(xg_table["xg_for"].mean()) if "xg_for" in xg_table else 1.35
    mean_xga = float(xg_table["xg_against"].mean()) if "xg_against" in xg_table else 1.25
    h_xg = _safe_float(hrow["xg_for"].iloc[0], mean_xg) if not hrow.empty else mean_xg
    h_xga = _safe_float(hrow["xg_against"].iloc[0], mean_xga) if not hrow.empty else mean_xga
    a_xg = _safe_float(arow["xg_for"].iloc[0], mean_xg) if not arow.empty else mean_xg
    a_xga = _safe_float(arow["xg_against"].iloc[0], mean_xga) if not arow.empty else mean_xga
    return h_xg, h_xga, a_xg, a_xga


def blend_expected_goals(
    goals_home_attack: float,
    goals_away_defense: float,
    goals_away_attack: float,
    goals_home_defense: float,
    xg_home_for: float,
    xg_home_against: float,
    xg_away_for: float,
    xg_away_against: float,
    xg_weight: float = 0.65,
    goals_weight: float = 0.35,
    league_home_mean: float = 1.45,
    league_away_mean: float = 1.20,
) -> Tuple[float, float]:
    lam_h_goals = (goals_home_attack * goals_away_defense) * league_home_mean
    lam_a_goals = (goals_away_attack * goals_home_defense) * league_away_mean
    lam_h_xg = (xg_home_for / max(xg_home_against, 0.1)) * league_home_mean * 0.9
    lam_a_xg = (xg_away_for / max(xg_away_against, 0.1)) * league_away_mean * 0.9
    lam_h_xg = max(0.4, min(3.5, lam_h_xg))
    lam_a_xg = max(0.4, min(3.5, lam_a_xg))
    lam_h_goals = max(0.4, min(3.5, lam_h_goals))
    lam_a_goals = max(0.4, min(3.5, lam_a_goals))
    w_xg = max(0.0, min(1.0, xg_weight))
    w_g = 1.0 - w_xg
    lam_h = w_xg * lam_h_xg + w_g * lam_h_goals
    lam_a = w_xg * lam_a_xg + w_g * lam_a_goals
    return round(lam_h, 3), round(lam_a, 3)
