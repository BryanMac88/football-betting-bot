"""
Optional Scrapling-powered enrichment for Paddy Power & BoyleSports.
The Odds API remains primary. This is a free fallback only.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from scrapling.fetchers import StealthyFetcher, Fetcher
    HAS_SCRAPLING = True
except ImportError:
    HAS_SCRAPLING = False


def _normalize(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"\b(fc|cf|afc|ac|sc|club|united|city)\b", "", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def scrape_paddypower_h2h(competition_url: str) -> List[Dict[str, Any]]:
    if not HAS_SCRAPLING:
        return []
    try:
        page = StealthyFetcher.fetch(
            competition_url,
            headless=True,
            network_idle=True,
            timeout=25000,
        )
        events = page.css("[data-testid*='event'], .event, .avb-event", adaptive=True)
        results = []
        for ev in events[:40]:
            text = ev.get_all_text(" ", strip=True)
            teams = re.findall(r"([A-Za-z0-9 .'\-]+)\s+v(?:s)?\s+([A-Za-z0-9 .'\-]+)", text)
            odds = re.findall(r"(\d+\.\d{2})", text)
            if teams and len(odds) >= 3:
                h, a = teams[0]
                results.append({
                    "home": h.strip(),
                    "away": a.strip(),
                    "home_norm": _normalize(h),
                    "away_norm": _normalize(a),
                    "home_odds": float(odds[0]),
                    "draw_odds": float(odds[1]),
                    "away_odds": float(odds[2]),
                    "book": "paddypower",
                    "source": "scrapling",
                })
        return results
    except Exception as e:
        logger.warning(f"Paddy Power scrape failed: {e}")
        return []


def scrape_boylesports_h2h(competition_url: str) -> List[Dict[str, Any]]:
    if not HAS_SCRAPLING:
        return []
    try:
        page = StealthyFetcher.fetch(
            competition_url,
            headless=True,
            network_idle=True,
            timeout=25000,
        )
        events = page.css(".event, [class*='event-'], .match", adaptive=True)
        results = []
        for ev in events[:40]:
            text = ev.get_all_text(" ", strip=True)
            teams = re.findall(r"([A-Za-z0-9 .'\-]+)\s+v(?:s)?\s+([A-Za-z0-9 .'\-]+)", text)
            odds = re.findall(r"(\d+\.\d{2})", text)
            if teams and len(odds) >= 3:
                h, a = teams[0]
                results.append({
                    "home": h.strip(),
                    "away": a.strip(),
                    "home_norm": _normalize(h),
                    "away_norm": _normalize(a),
                    "home_odds": float(odds[0]),
                    "draw_odds": float(odds[1]),
                    "away_odds": float(odds[2]),
                    "book": "boylesports",
                    "source": "scrapling",
                })
        return results
    except Exception as e:
        logger.warning(f"BoyleSports scrape failed: {e}")
        return []


def enrich_named_odds(
    fixtures: List[Dict[str, str]],
    pp_urls: Optional[Dict[str, str]] = None,
    bs_urls: Optional[Dict[str, str]] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not HAS_SCRAPLING:
        return {}
    default_pp = {
        "PL": "https://www.paddypower.com/football/english-premier-league",
        "PD": "https://www.paddypower.com/football/spanish-la-liga",
        "SA": "https://www.paddypower.com/football/italian-serie-a",
        "BL1": "https://www.paddypower.com/football/german-bundesliga",
        "FL1": "https://www.paddypower.com/football/french-ligue-1",
        "ELC": "https://www.paddypower.com/football/english-championship",
    }
    pp_urls = pp_urls or default_pp
    enriched = {}
    needed = {f.get("league") for f in fixtures if f.get("league")}
    for league in needed:
        url = pp_urls.get(league)
        if url:
            for row in scrape_paddypower_h2h(url):
                key = (row["home_norm"], row["away_norm"])
                enriched[key] = {
                    "home": row["home_odds"],
                    "draw": row["draw_odds"],
                    "away": row["away_odds"],
                    "book": "paddypower",
                }
    return enriched
