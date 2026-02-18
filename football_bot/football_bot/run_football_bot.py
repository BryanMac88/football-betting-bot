import os
import json
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ================= CONFIG =================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

STATUS_TAB = "FOOTBALL_STATUS"
MATCHES_TAB = "FOOTBALL_MATCHES"

ODDS_RAW_TAB = "FOOTBALL_ODDS_RAW"
ODDS_TAB = "FOOTBALL_ODDS"

COMPETITIONS = ["PL"]         # keep tight for free tiers
LOOKBACK_DAYS = 365
LOOKAHEAD_DAYS = 7

BASE_URL = "https://api.football-data.org/v4"

# ---- ODDS API ----
# Your secret is ODDS_API_KEY. Provider endpoint differs by service.
# We'll support two common patterns:
# 1) The Odds API (the-odds-api.com) v4
# 2) odds-api.io v3 (placeholder)
ODDS_PROVIDER = os.getenv("ODDS_PROVIDER", "theoddsapi")  # "theoddsapi" or "oddsapiio"
# =========================================


def append_status(sheet, message: str):
    try:
        ws = sheet.worksheet(STATUS_TAB)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=STATUS_TAB, rows=200, cols=10)
        ws.append_row(["timestamp_utc", "message"])
    ws.append_row([datetime.now(timezone.utc).isoformat(), message])


def upsert_df(sheet, tab_name: str, df: pd.DataFrame, rows=2000, cols=30):
    try:
        ws = sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=tab_name, rows=rows, cols=cols)

    if df is None or df.empty:
        ws.clear()
        ws.update([["no data"]])
        return

    values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
    ws.clear()
    ws.update(values)


def get_headers_football_data():
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN")
    return {"X-Auth-Token": token}


def fetch_matches(comp, date_from, date_to) -> pd.DataFrame:
    url = f"{BASE_URL}/competitions/{comp}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to}
    r = requests.get(url, headers=get_headers_football_data(), params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["matches"]

    rows = []
    for m in data:
        rows.append({
            "competition": comp,
            "utcDate": m["utcDate"],
            "status": m["status"],
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "match_id": m["id"],
        })
    return pd.DataFrame(rows)


def implied_prob(dec_odds: float):
    try:
        o = float(dec_odds)
        if o <= 1.0:
            return None
        return 1.0 / o
    except Exception:
        return None


def normalize_3way(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return (None, None, None)
    s = p1 + p2 + p3
    if s <= 0:
        return (None, None, None)
    return (p1 / s, p2 / s, p3 / s)


def normalize_2way(p1, p2):
    if p1 is None or p2 is None:
        return (None, None)
    s = p1 + p2
    if s <= 0:
        return (None, None)
    return (p1 / s, p2 / s)


# ================= ODDS FETCHERS =================

def fetch_odds_theoddsapi():
    """
    The Odds API v4 example for soccer:
    https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?regions=eu&markets=h2h,totals,btts&oddsFormat=decimal&apiKey=...
    """
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY")

    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    params = {
        "apiKey": key,
        "regions": "eu",
        "markets": "h2h,totals,btts",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()  # list of events


def fetch_odds_oddsapiio():
    # Placeholder; structure varies. We still log raw for later parsing.
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY")
    url = "https://api.odds-api.io/v3/events"
    params = {"apiKey": key, "sport": "football"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_provider_raw():
    if ODDS_PROVIDER.lower() == "theoddsapi":
        return fetch_odds_theoddsapi(), "theoddsapi"
    if ODDS_PROVIDER.lower() == "oddsapiio":
        return fetch_odds_oddsapiio(), "oddsapiio"
    raise RuntimeError(f"Unknown ODDS_PROVIDER: {ODDS_PROVIDER}")


def parse_theoddsapi_events(raw_events):
    """
    Extract best-available (max odds) across books for:
      - 1X2 (h2h)
      - O/U 2.5 (totals with point=2.5)
      - BTTS (btts yes/no)
    """
    rows = []
    for ev in raw_events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        event_id = ev.get("id")

        best = {
            "odds_1x2_home": None,
            "odds_1x2_away": None,
            "odds_1x2_draw": None,
            "odds_ou25_over": None,
            "odds_ou25_under": None,
            "odds_btts_yes": None,
            "odds_btts_no": None,
        }

        for book in ev.get("bookmakers", []):
            for mkt in book.get("markets", []):
                key = mkt.get("key")

                # 1X2
                if key == "h2h":
                    for out in mkt.get("outcomes", []):
                        name = out.get("name")
                        price = out.get("price")
                        if name == home:
                            best["odds_1x2_home"] = max(best["odds_1x2_home"] or 0, price or 0) or best["odds_1x2_home"]
                        elif name == away:
                            best["odds_1x2_away"] = max(best["odds_1x2_away"] or 0, price or 0) or best["odds_1x2_away"]
                        elif name in ("Draw", "draw"):
                            best["odds_1x2_draw"] = max(best["odds_1x2_draw"] or 0, price or 0) or best["odds_1x2_draw"]

                # Totals
                if key == "totals":
                    point = mkt.get("outcomes", [{}])[0].get("point")
                    # outcomes usually have point per outcome; check each
                    for out in mkt.get("outcomes", []):
                        if out.get("point") != 2.5:
                            continue
                        name = out.get("name")  # Over / Under
                        price = out.get("price")
                        if name == "Over":
                            best["odds_ou25_over"] = max(best["odds_ou25_over"] or 0, price or 0) or best["odds_ou25_over"]
                        elif name == "Under":
                            best["odds_ou25_under"] = max(best["odds_ou25_under"] or 0, price or 0) or best["odds_ou25_under"]

                # BTTS
                if key == "btts":
                    for out in mkt.get("outcomes", []):
                        name = out.get("name")  # Yes/No
                        price = out.get("price")
                        if name == "Yes":
                            best["odds_btts_yes"] = max(best["odds_btts_yes"] or 0, price or 0) or best["odds_btts_yes"]
                        elif name == "No":
                            best["odds_btts_no"] = max(best["odds_btts_no"] or 0, price or 0) or best["odds_btts_no"]

        # Fair probs
        pH_raw = implied_prob(best["odds_1x2_home"])
        pD_raw = implied_prob(best["odds_1x2_draw"])
        pA_raw = implied_prob(best["odds_1x2_away"])
        pH, pD, pA = normalize_3way(pH_raw, pD_raw, pA_raw)

        pOv_raw = implied_prob(best["odds_ou25_over"])
        pUn_raw = implied_prob(best["odds_ou25_under"])
        pOv, pUn = normalize_2way(pOv_raw, pUn_raw)

        pYes_raw = implied_prob(best["odds_btts_yes"])
        pNo_raw = implied_prob(best["odds_btts_no"])
        pYes, pNo = normalize_2way(pYes_raw, pNo_raw)

        rows.append({
            "event_id": event_id,
            "commence_time": commence,
            "home": home,
            "away": away,
            **best,
            "p_mkt_home": pH,
            "p_mkt_draw": pD,
            "p_mkt_away": pA,
            "p_mkt_ou_over": pOv,
            "p_mkt_ou_under": pUn,
            "p_mkt_btts_yes": pYes,
            "p_mkt_btts_no": pNo,
        })

    return pd.DataFrame(rows)


def main():
    sheet_id = os.getenv("SHEET_ID")
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sheet_id:
        raise RuntimeError("Missing SHEET_ID")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # ---- Matches (keep) ----
    today = datetime.now(timezone.utc).date()
    date_from = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    date_to = (today + timedelta(days=LOOKAHEAD_DAYS)).isoformat()

    match_frames = [fetch_matches(comp, date_from, date_to) for comp in COMPETITIONS]
    matches_df = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    upsert_df(sh, MATCHES_TAB, matches_df)
    append_status(sh, f"Loaded {len(matches_df)} matches from football-data")

    # ---- Odds (new) ----
    raw, provider = pick_provider_raw()

    # Log ONE raw sample so we can debug structure if needed
    sample = None
    if provider == "theoddsapi":
        sample = raw[0] if isinstance(raw, list) and len(raw) else None
    elif provider == "oddsapiio":
        # odds-api.io returns dict; try first event if present
        data = raw.get("data") if isinstance(raw, dict) else None
        sample = data[0] if isinstance(data, list) and len(data) else raw

    raw_df = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "raw_sample_json": json.dumps(sample)[:49000] if sample is not None else "",
    }])
    upsert_df(sh, ODDS_RAW_TAB, raw_df, rows=200, cols=5)
    append_status(sh, f"Saved raw odds sample ({provider}) to {ODDS_RAW_TAB}")

    # Parse normalized odds (only implemented for The Odds API for now)
    if provider == "theoddsapi":
        odds_df = parse_theoddsapi_events(raw)
        upsert_df(sh, ODDS_TAB, odds_df, rows=2000, cols=40)
        append_status(sh, f"Parsed {len(odds_df)} odds events into {ODDS_TAB}")
    else:
        upsert_df(sh, ODDS_TAB, pd.DataFrame([{"message": "Parser not implemented for this provider yet. See FOOTBALL_ODDS_RAW."}]))
        append_status(sh, "Odds parser not implemented for provider; logged raw only")

    print("Step 6 complete.")


if __name__ == "__main__":
    main()
