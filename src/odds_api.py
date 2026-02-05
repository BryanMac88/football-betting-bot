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
