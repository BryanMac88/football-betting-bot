from __future__ import annotations
import requests
from typing import Any, Iterable, Optional
import pandas as pd

BASE = "https://api.the-odds-api.com/v4"

class OddsApi:
    def __init__(self, api_key: str, region: str = "uk"):
        self.api_key = api_key
        self.region = region

    def list_sports(self) -> list[dict[str, Any]]:
        r = requests.get(f"{BASE}/sports", params={"apiKey": self.api_key})
        r.raise_for_status()
        return r.json()

    def get_odds(self, sport_key: str, markets: Iterable[str]) -> list[dict[str, Any]]:
        # Pull best odds across bookmakers for each market
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        r = requests.get(f"{BASE}/sports/{sport_key}/odds", params=params)
        r.raise_for_status()
        return r.json()

def flatten_odds(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        sport_key = ev.get("sport_key")
        sport_title = ev.get("sport_title")
        for bk in ev.get("bookmakers", []):
            bk_key = bk.get("key")
            bk_title = bk.get("title")
            last_update = bk.get("last_update")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                for out in m.get("outcomes", []):
                    rows.append({
                        "sport_key": sport_key,
                        "competition": sport_title,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker_key": bk_key,
                        "bookmaker": bk_title,
                        "last_update": last_update,
                        "market": mkey,
                        "selection": out.get("name"),
                        "price": out.get("price"),
                        "point": out.get("point"),  # totals/cards line
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep best price per event/market/selection/point across bookmakers
    grp_cols = ["competition","commence_time","home_team","away_team","market","selection","point"]
    df_best = (
        df.sort_values("price", ascending=False)
          .groupby(grp_cols, as_index=False)
          .first()
    )
    return df_best


# ================================================================
# Blending model probabilities with market odds, and matching
# football-data.org fixtures to Odds API events.
# ================================================================
import difflib
import re
from datetime import datetime
from typing import Dict, List, Tuple

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


def best_h2h_prices(event: Dict[str, Any]) -> Dict[str, float]:
    """Best (highest) h2h price per outcome across ALL bookmakers."""
    best: Dict[str, float] = {}
    for bm in event.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            for outcome in mkt.get("outcomes", []):
                name = outcome["name"]
                price = float(outcome["price"])
                if name not in best or price > best[name]:
                    best[name] = price
    return best


# ---- Named-bookmaker odds (Paddy Power / Boylesports) ----
# The odds filter should reflect what YOU would actually be offered at
# your own bookmaker, not a best-across-all-books blend that may not be
# available to you. PREFERRED_BOOKMAKERS is checked in order.
#
# IMPORTANT: The Odds API's coverage of these two specific books is not
# guaranteed — run check_bookmakers.py to see which are actually live
# for your key. ODDS_FILTER_REQUIRE_NAMED_BOOK in main.py controls what
# happens when neither is available for a fixture.
PREFERRED_BOOKMAKERS = ["paddypower", "boylesports"]


def named_h2h_prices(
    event: Dict[str, Any], preferred: List[str] = None
) -> Tuple[Dict[str, float], Optional[str]]:
    """Returns (prices, bookmaker_key_used). First preferred bookmaker
    that has an h2h market for this event wins. Returns ({}, None) if
    none of them cover this event."""
    preferred = preferred or PREFERRED_BOOKMAKERS
    for want in preferred:
        for bm in event.get("bookmakers", []):
            if bm.get("key") != want:
                continue
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                prices = {o["name"]: float(o["price"]) for o in mkt.get("outcomes", [])}
                if prices:
                    return prices, bm.get("key")
    return {}, None


def named_totals_prices(
    event: Dict[str, Any], preferred: List[str] = None
) -> Tuple[Dict[float, Dict[str, float]], Optional[str]]:
    """Returns ({line: {'Over': price, 'Under': price}}, bookmaker_key)
    for the totals (over/under goals) market."""
    preferred = preferred or PREFERRED_BOOKMAKERS
    for want in preferred:
        for bm in event.get("bookmakers", []):
            if bm.get("key") != want:
                continue
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                by_line: Dict[float, Dict[str, float]] = {}
                for o in mkt.get("outcomes", []):
                    point = o.get("point")
                    if point is None:
                        continue
                    by_line.setdefault(float(point), {})[o["name"]] = float(o["price"])
                if by_line:
                    return by_line, bm.get("key")
    return {}, None


def all_totals_prices(event: Dict[str, Any]) -> Dict[float, Dict[str, float]]:
    """Best totals price per line/side across ALL bookmakers (fallback)."""
    best: Dict[float, Dict[str, float]] = {}
    for bm in event.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "totals":
                continue
            for o in mkt.get("outcomes", []):
                point = o.get("point")
                if point is None:
                    continue
                line = float(point)
                name = o["name"]
                price = float(o["price"])
                cur = best.setdefault(line, {})
                if name not in cur or price > cur[name]:
                    cur[name] = price
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
    normalized team names + kickoff proximity. Returns None rather than
    guessing if nothing scores highly enough."""
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
