from __future__ import annotations
from io import StringIO

import requests
import pandas as pd


def read_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # football-data.co.uk uses ISO-8859-1 sometimes
    content = r.content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("ISO-8859-1")
    # on_bad_lines="skip": football-data CSVs occasionally contain malformed rows
    df = pd.read_csv(StringIO(text), on_bad_lines="skip")
    return df


def normalize_results(df: pd.DataFrame, league_code: str) -> pd.DataFrame:
    # Expected common columns:
    # Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HF, AF, HY, AY, HR, AR, etc.
    out = df.copy()
    out["league_code"] = league_code
    # Parse date (varies)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    # Rename common goal cols
    rename = {}
    if "FTHG" in out.columns: rename["FTHG"] = "home_goals"
    if "FTAG" in out.columns: rename["FTAG"] = "away_goals"
    if "HomeTeam" in out.columns: rename["HomeTeam"] = "home_team"
    if "AwayTeam" in out.columns: rename["AwayTeam"] = "away_team"
    out = out.rename(columns=rename)
    # Cards & fouls (if present)
    if "HF" in out.columns: out = out.rename(columns={"HF": "home_fouls"})
    if "AF" in out.columns: out = out.rename(columns={"AF": "away_fouls"})
    if "HY" in out.columns: out = out.rename(columns={"HY": "home_yellows"})
    if "AY" in out.columns: out = out.rename(columns={"AY": "away_yellows"})
    if "HR" in out.columns: out = out.rename(columns={"HR": "home_reds"})
    if "AR" in out.columns: out = out.rename(columns={"AR": "away_reds"})
    keep = ["league_code", "Date", "home_team", "away_team", "home_goals", "away_goals",
            "home_fouls", "away_fouls", "home_yellows", "away_yellows", "home_reds", "away_reds"]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    return out[keep]


def get_blend_weights(league_code: str, cfg: dict) -> tuple[float, float]:
    """Return (xg_weight, goals_weight) for a league.

    Leagues with no xG source (League One, League Two, National League)
    are listed under "goals_only_leagues" in the config and use goals only.
    """
    if league_code in cfg.get("goals_only_leagues", []):
        return 0.0, 1.0
    return cfg.get("xg_weight", 0.65), cfg.get("goals_weight", 0.35)
