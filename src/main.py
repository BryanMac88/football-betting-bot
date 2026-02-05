from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable, Any

import pandas as pd

# --- existing imports you already had ---
# We keep these names because your repo already has them.
# If any of these imports fail in YOUR repo, tell me and I’ll adjust.
from .utils import env, utc_now_iso, load_json  # type: ignore
from .odds_api import OddsApi  # type: ignore
from .sheets import write_df, append_log  # type: ignore
from .pipelines import (  # type: ignore
    pick_target_sports,
    flatten_odds,
    normalize_results,
    build_team_form,
    build_predictions,
    build_value_bets,
)

# ----------------------------
# Logging helpers
# ----------------------------

def _log(msg: str) -> None:
    # Always prints to GitHub Actions
    print(msg, flush=True)

def _df_info(name: str, df: pd.DataFrame | None) -> None:
    if df is None:
        _log(f"{name}: None")
        return
    _log(f"{name}: rows={len(df):,} cols={len(df.columns)}")
    if len(df) > 0:
        _log(f"{name}: columns={list(df.columns)[:25]}{'...' if len(df.columns) > 25 else ''}")

def _safe_write(sheet_name: str, df: pd.DataFrame) -> None:
    """Always writes a DataFrame (even empty) so the tab exists and has headers."""
    _log(f"Writing sheet '{sheet_name}' ...")
    _df_info(f"{sheet_name}", df)
    write_df(sheet_name, df)  # existing helper in your repo
    _log(f"✅ Wrote '{sheet_name}'")

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    run_ts = utc_now_iso()
    _log(f"=== Football bot run starting: {run_ts} ===")

    # Required env (from GitHub secrets)
    sheet_id = env("SHEET_ID")
    odds_key = env("ODDS_API_KEY")

    region = os.getenv("REGION", "uk")
    bankroll = float(os.getenv("BANKROLL", "100"))
    kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.25"))
    min_edge = float(os.getenv("MIN_EDGE", "0.03"))

    _log(f"Config: REGION={region} BANKROLL={bankroll} KELLY_FRACTION={kelly_fraction} MIN_EDGE={min_edge}")

    # Init API client
    odds = OddsApi(api_key=odds_key, region=region)

    # 1) Pick sports (competitions)
    sports = odds.list_sports()
    _log(f"Odds API sports returned: {len(sports):,}")
    target_sports = pick_target_sports(sports)
    _log(f"Target sports matched: {len(target_sports):,}")
    if len(target_sports) == 0:
        _log("❌ No target sports matched. This is why everything is empty.")
        append_log({
            "ts": run_ts,
            "level": "ERROR",
            "msg": "No target sports matched from Odds API sports list",
        })
        # still write empty tabs so you can see structure
        _safe_write("Odds_Snapshot", pd.DataFrame())
        _safe_write("Fixtures", pd.DataFrame())
        _safe_write("Model_Probs", pd.DataFrame())
        _safe_write("Value_Bets", pd.DataFrame())
        return

    # 2) Fetch odds/events for each sport
    odds_rows: list[pd.DataFrame] = []
    fixtures_rows: list[pd.DataFrame] = []

    for s in target_sports:
        key = s.get("key") if isinstance(s, dict) else getattr(s, "key", None)
        title = s.get("title") if isinstance(s, dict) else getattr(s, "title", "")
        if not key:
            continue

        _log(f"Fetching odds for: {title} ({key})")
        try:
            events = odds.get_odds_for_sport(key=key)  # your existing method
        except Exception as e:
            _log(f"❌ Odds fetch failed for {key}: {e}")
            continue

        if events is None:
            _log(f"{key}: events=None")
            continue

        # Your repo likely returns dict/json; flatten it
        try:
            flat = flatten_odds(events, sport_key=key)
        except Exception as e:
            _log(f"❌ flatten_odds failed for {key}: {e}")
            continue

        if isinstance(flat, pd.DataFrame):
            odds_rows.append(flat)
            _log(f"{key}: odds rows={len(flat):,}")
        else:
            _log(f"{key}: flatten_odds did not return DataFrame")

        # Fixtures (if your flatten_odds already contains commence_time/home/away, we can reuse)
        # Keep it simple: fixtures derived from odds DataFrame if possible
        if isinstance(flat, pd.DataFrame) and {"commence_time", "home_team", "away_team"}.issubset(set(flat.columns)):
            fx = (
                flat[["sport_key", "sport_title", "commence_time", "home_team", "away_team"]]
                .drop_duplicates()
                .sort_values(["commence_time", "sport_title"], ascending=[True, True])
                .reset_index(drop=True)
            )
            fixtures_rows.append(fx)

    odds_df = pd.concat(odds_rows, ignore_index=True) if odds_rows else pd.DataFrame()
    fixtures_df = pd.concat(fixtures_rows, ignore_index=True) if fixtures_rows else pd.DataFrame()

    _df_info("Odds_Snapshot df", odds_df)
    _df_info("Fixtures df", fixtures_df)

    _safe_write("Odds_Snapshot", odds_df)
    _safe_write("Fixtures", fixtures_df)

    if odds_df.empty:
        _log("❌ Odds_Snapshot is empty. No value bets can be computed.")
        append_log({"ts": run_ts, "level": "WARN", "msg": "Odds_Snapshot empty (0 rows). Check target leagues/markets/region/quota."})
        _safe_write("Model_Probs", pd.DataFrame())
        _safe_write("Value_Bets", pd.DataFrame())
        return

    # 3) Build team form + predictions
    try:
        results_df = normalize_results()  # if your pipeline pulls free CSVs internally
    except Exception as e:
        _log(f"❌ normalize_results failed: {e}")
        results_df = pd.DataFrame()

    _df_info("Results df", results_df)

    try:
        team_form = build_team_form(results_df)
    except Exception as e:
        _log(f"❌ build_team_form failed: {e}")
        team_form = pd.DataFrame()

    _df_info("Team form df", team_form)

    try:
        probs_df = build_predictions(fixtures_df, team_form)
    except Exception as e:
        _log(f"❌ build_predictions failed: {e}")
        probs_df = pd.DataFrame()

    _df_info("Model_Probs df", probs_df)
    _safe_write("Model_Probs", probs_df)

    # 4) Value bets
    try:
        value_df = build_value_bets(
            odds_df=odds_df,
            probs_df=probs_df,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge,
        )
    except Exception as e:
        _log(f"❌ build_value_bets failed: {e}")
        value_df = pd.DataFrame()

    _df_info("Value_Bets df", value_df)
    _safe_write("Value_Bets", value_df)

    # 5) Always append a run log row
    append_log({
        "ts": run_ts,
        "level": "INFO",
        "msg": "Run completed",
        "sports_total": int(len(sports)),
        "sports_target": int(len(target_sports)),
        "odds_rows": int(len(odds_df)),
        "fixtures_rows": int(len(fixtures_df)),
        "results_rows": int(len(results_df)) if isinstance(results_df, pd.DataFrame) else 0,
        "probs_rows": int(len(probs_df)) if isinstance(probs_df, pd.DataFrame) else 0,
        "value_rows": int(len(value_df)) if isinstance(value_df, pd.DataFrame) else 0,
    })

    _log("=== Football bot run finished ===")
