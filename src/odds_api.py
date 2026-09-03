"""
The Odds API integration.

Fetches bookmaker odds, de-vigs them into fair (overround-removed)
probabilities, and matches fixtures to football-data.org fixtures by
team name + kickoff time so they can be blended with the Poisson model
in main.py.

VERIFY BEFORE RELYING ON THIS: the FD_TO_ODDS_SPORT mapping below is
best-effort. Confirm exact sport keys for your account by calling:
    GET https://api.the-odds-api.com/v4/sports?apiKey=YOUR_KEY
and adjust the mapping to match what's actually returned — Odds API
coverage and key names can differ by plan/region and do change over time.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Best-effort mapping from football-data.org competition codes to
# The Odds API sport keys. CONFIRM against /v4/sports before trusting.
FD_TO_ODDS_SPORT = {
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "SA":  "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "CL":  "soccer_uefa_champs_league",
    "EL":  "soccer_uefa_europa_league",
    "ELC": "soccer_efl_champ",
    "EL1": "soccer_england_league1",
    "EL2": "soccer_england_league2",
    "SD":  "soccer_spain_segunda_division",
    # SPL (Scottish Premiership) and others: add once confirmed against
    # the /v4/sports response — leaving unmapped competitions out is
    # safe, they'll just be skipped for odds blending (model-only).
}

TEAM_STOPWORDS = re.compile(
    r"\b(fc|cf|afc|ac|sc|club|calcio|cd|ud|rc|as|ss|ssc)\b", re.IGNORECASE
)


def normalize_team(name: str) -> str:
    n = (name or "").lower()
    n = TEAM_STOPWORDS.sub("", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


@dataclass
class OddsAPI:
    api_key: str
    region: str = "uk"
    markets: str = "h2h,totals"  # btts/alternate_totals_cards support is patchy across UK books
    base: str = ODDS_API_BASE

    def get_odds(self, sport_key: str) -> List[Dict[str, Any]]:
        r = requests.get(
            f"{self.base}/sports/{sport_key}/odds",
            params={
                "apiKey": self.api_key,
                "regions": self.region,
                "markets": self.markets,
                "oddsFormat": "decimal",
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


def devig_three_way(odds_h: float, odds_d: float, odds_a: float) -> Tuple[float, float, float]:
    """Convert three decimal odds into fair probabilities by removing
    the bookmaker's overround (margin) via simple normalization."""
    ih, idr, ia = 1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a
    total = ih + idr + ia
    return ih / total, idr / total, ia / total


def devig_two_way(odds_a: float, odds_b: float) -> Tuple[float, float]:
    ia, ib = 1.0 / odds_a, 1.0 / odds_b
    total = ia + ib
    return ia / total, ib / total


def best_prices(event: Dict[str, Any], market_key: str) -> Dict[str, float]:
    """Best (highest) price per outcome across all bookmakers in the
    response. Using the best available price per outcome, then de-vigging
    that combined line, is a common way to build a fair consensus
    probability without being biased by any single bookmaker's margin."""
    best: Dict[str, float] = {}
    for bm in event.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            for outcome in mkt.get("outcomes", []):
                name = outcome["name"]
                price = float(outcome["price"])
                if name not in best or price > best[name]:
                    best[name] = price
    return best


def match_fixture(
    fd_home: str,
    fd_away: str,
    fd_kickoff: str,
    odds_events: List[Dict[str, Any]],
    max_hours_diff: float = 6.0,
    min_match_score: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """Fuzzy-match a football-data.org fixture to an Odds API event by
    normalized team names + kickoff proximity. Returns None (no blend
    for this fixture) rather than guessing, if nothing scores highly
    enough — a wrong match would silently corrupt the model."""
    try:
        fd_dt = datetime.fromisoformat(fd_kickoff.replace("Z", "+00:00"))
    except Exception:
        return None

    nh, na = normalize_team(fd_home), normalize_team(fd_away)
    best_event, best_score = None, 0.0

    for ev in odds_events:
        try:
            ev_dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        if abs((ev_dt - fd_dt).total_seconds()) > max_hours_diff * 3600:
            continue

        eh = normalize_team(ev.get("home_team", ""))
        ea = normalize_team(ev.get("away_team", ""))
        score = (
            difflib.SequenceMatcher(None, nh, eh).ratio()
            + difflib.SequenceMatcher(None, na, ea).ratio()
        )
        if score > best_score:
            best_score, best_event = score, ev

    if best_event is not None and best_score >= min_match_score:
        return best_event
    return None
