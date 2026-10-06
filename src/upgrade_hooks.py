"""
Drop-in helpers so the existing main.py can adopt the free xG + Scrapling upgrades
with minimal changes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from xg_data import build_xg_team_strength
    HAS_XG = True
except ImportError:
    HAS_XG = False


def load_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "leagues.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_xg_tables(league_codes: List[str]) -> Dict[str, pd.DataFrame]:
    if not HAS_XG:
        logger.warning("xg_data not available – returning empty xG tables")
        return {}
    cfg = load_config()
    fbref_map = cfg.get("fbref_leagues", {})
    tables = {}
    for code in league_codes:
        try:
            fbref_key = fbref_map.get(code)
            df = build_xg_team_strength(code, fbref_key=fbref_key)
            if not df.empty:
                tables[code] = df
                logger.info(f"xG loaded for {code}: {len(df)} teams ({df['source'].iloc[0]})")
            else:
                logger.info(f"No xG data for {code}")
        except Exception as e:
            logger.warning(f"xG load failed for {code}: {e}")
    return tables


def get_model_params():
    from model import ModelParams
    cfg = load_config()
    return ModelParams(
        rolling_window_matches=int(cfg.get("rolling_window_matches", 12)),
        xg_weight=float(cfg.get("xg_weight", 0.65)),
        goals_weight=float(cfg.get("goals_weight", 0.35)),
    )
