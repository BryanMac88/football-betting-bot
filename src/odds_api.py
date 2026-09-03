from __future__ import annotations
import requests
from typing import Any, Iterable
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
# ADDED: pieces needed to blend model probabilities with market odds
# and to match football-data.org fixtures to Odds API events.
# ================================================================
import difflib
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Best-effort mapping from football-data.org competition codes to
# The Odds API sport keys. VERIFY against OddsApi(...).list_sports()
# before relying on this — coverage and key names can change.
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
    # SPL (Scottish Premiership) and others: add once confirmed via
    # list_sports() — leaving unmapped competitions out just skips
    # odds blending for them (falls back to model-only).
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
    """Best (highest) h2h price per outcome across bookmakers, taken
    directly from a raw Odds API event (before flattening)."""
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
