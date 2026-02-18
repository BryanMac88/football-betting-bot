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

COMPETITIONS = ["PL"]
LOOKBACK_DAYS = 365
LOOKAHEAD_DAYS = 7

BASE_URL = "https://api.football-data.org/v4"

# Odds provider selector
ODDS_PROVIDER = os.getenv("ODDS_PROVIDER", "theoddsapi").strip().lower()
# =========================================


# ================= SHEETS HELPERS =================

def append_status(sheet, message: str):
    try:
        ws = sheet.worksheet(STATUS_TAB)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=STATUS_TAB, rows=300, cols=10)
        ws.append_row(["timestamp_utc", "message"])
    ws.append_row([datetime.now(timezone.utc).isoformat(), message])


def upsert_df(sheet, tab_name: str, df: pd.DataFrame, rows: int = 2000, cols: int = 50):
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


# ================= FOOTBALL-DATA.ORG FETCH =================

def get_headers_football_data():
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN")
    return {"X-Auth-Token": token}


def fetch_matches(comp: str, date_from: str, date_to: str) -> pd.DataFrame:
    url = f"{BASE_URL}/competitions/{comp}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to}
    r = requests.get(url, headers=get_headers_football_data(), params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("matches", [])

    rows = []
    for m in data:
        rows.append({
            "competition": comp,
            "utcDate": m.get("utcDate"),
            "status": m.get("status"),
            "home": (m.get("homeTeam") or {}).get("name"),
            "away": (m.get("awayTeam") or {}).get("name"),
            "home_goals": ((m.get("score") or {}).get("fullTime") or {}).get("home"),
            "away_goals": ((m.get("score") or {}).get("fullTime") or {}).get("away"),
            "match_id": m.get("id"),
        })
    return pd.DataFrame(rows)


# ================= ODDS MATH HELPERS =================

def implied_prob(dec_odds):
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

def fetch_odds_theoddsapi_with_fallback():
    """
    The Odds API v4 with a fallback ladder.
    Instead of raising on 422, we try simpler requests.
    Returns (events_list, used_params, debug_message).
    """
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY")

    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"

    attempts = [
        # Richest
        {"apiKey": key, "regions": "eu", "markets": "h2h,totals,btts", "oddsFormat": "decimal", "dateFormat": "iso"},
        {"apiKey": key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"},
        {"apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"},
        # Alternative region
        {"apiKey": key, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"},
        # Minimal
        {"apiKey": key},
    ]

    last_debug = None
    for params in attempts:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if not isinstance(data, list):
                last_debug = f"200 OK but payload not a list. First 300 chars: {str(r.text)[:300]}"
                continue
            return data, params, f"OK using params: {params}"

        # keep the API message so you can see what it rejects
        last_debug = f"Status {r.status_code} for params={params}. Body: {r.text[:900]}"

    # If we got here, nothing worked
    return [], {}, f"All attempts failed. Last: {last_debug}"


def fetch_odds_oddsapiio_raw():
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY")
    url = "https://api.odds-api.io/v3/events"
    params = {"apiKey": key, "sport": "football"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_provider_raw():
    if ODDS_PROVIDER == "theoddsapi":
        events, used_params, debug = fetch_odds_theoddsapi_with_fallback()
        return events, "theoddsapi", used_params, debug
    if ODDS_PROVIDER == "oddsapiio":
        raw = fetch_odds_oddsapiio_raw()
        return raw, "oddsapiio", {}, "OK (oddsapiio raw)"
    raise RuntimeError(f"Unknown ODDS_PROVIDER: {ODDS_PROVIDER}")


# ================= ODDS PARSER (The Odds API) =================

def parse_theoddsapi_events(raw_events):
    """
    Extract best available (max odds across books) for:
      - 1X2 (h2h)
      - O/U 2.5 (totals, point=2.5) [optional]
      - BTTS (btts yes/no)          [optional]
    """
    if not isinstance(raw_events, list) or not raw_events:
        return pd.DataFrame()

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

        for book in ev.get("bookmakers", []) or []:
            for mkt in book.get("markets", []) or []:
                mkey = mkt.get("key")

                if mkey == "h2h":
                    for out in mkt.get("outcomes", []) or []:
                        name = out.get("name")
                        price = out.get("price")
                        if price is None:
                            continue
                        if name == home:
                            best["odds_1x2_home"] = max(best["odds_1x2_home"] or 0, price) or best["odds_1x2_home"]
                        elif name == away:
                            best["odds_1x2_away"] = max(best["odds_1x2_away"] or 0, price) or best["odds_1x2_away"]
                        elif str(name).lower() == "draw":
                            best["odds_1x2_draw"] = max(best["odds_1x2_draw"] or 0, price) or best["odds_1x2_draw"]

                if mkey == "totals":
                    for out in mkt.get("outcomes", []) or []:
                        if out.get("point") != 2.5:
                            continue
                        name = out.get("name")  # Over/Under
                        price = out.get("price")
                        if price is None:
                            continue
                        if name == "Over":
                            best["odds_ou25_over"] = max(best["odds_ou25_over"] or 0, price) or best["odds_ou25_over"]
                        elif name == "Under":
                            best["odds_ou25_under"] = max(best["odds_ou25_under"] or 0, price) or best["odds_ou25_under"]

                if mkey == "btts":
                    for out in mkt.get("outcomes", []) or []:
                        name = out.get("name")  # Yes/No
                        price = out.get("price")
                        if price is None:
                            continue
                        if name == "Yes":
                            best["odds_btts_yes"] = max(best["odds_btts_yes"] or 0, price) or best["odds_btts_yes"]
                        elif name == "No":
                            best["odds_btts_no"] = max(best["odds_btts_no"] or 0, price) or best["odds_btts_no"]

        # Fair (overround removed) probabilities
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


# ================= MAIN =================

def main():
    # Sheets auth
    sheet_id = os.getenv("SHEET_ID")
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sheet_id:
        raise RuntimeError("Missing SHEET_ID")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # Matches
    today = datetime.now(timezone.utc).date()
    date_from = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    date_to = (today + timedelta(days=LOOKAHEAD_DAYS)).isoformat()

    match_frames = [fetch_matches(comp, date_from, date_to) for comp in COMPETITIONS]
    matches_df = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    upsert_df(sh, MATCHES_TAB, matches_df, rows=2500, cols=20)
    append_status(sh, f"Loaded {len(matches_df)} matches from football-data")

    # Odds (won't crash on 422 now)
    raw, provider, used_params, debug_msg = pick_provider_raw()
    append_status(sh, f"Odds fetch provider={provider}. {debug_msg}")

    # Always store a raw sample or error details
    if provider == "theoddsapi":
        sample = raw[0] if isinstance(raw, list) and len(raw) else {"error": debug_msg, "used_params": used_params}
    else:
        # oddsapiio returns dict
        sample = raw if isinstance(raw, dict) else {"raw_type": str(type(raw))}

    raw_df = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "used_params": json.dumps(used_params)[:20000],
        "raw_sample_json": json.dumps(sample)[:49000],
    }])
    upsert_df(sh, ODDS_RAW_TAB, raw_df, rows=200, cols=10)

    # Parse if theoddsapi and we actually got events
    if provider == "theoddsapi":
        odds_df = parse_theoddsapi_events(raw)
        if odds_df.empty:
            upsert_df(sh, ODDS_TAB, pd.DataFrame([{
                "message": "No odds events parsed. See FOOTBALL_ODDS_RAW for API error details."
            }]), rows=50, cols=5)
            append_status(sh, "FOOTBALL_ODDS not populated (no events). Check FOOTBALL_ODDS_RAW.")
        else:
            upsert_df(sh, ODDS_TAB, odds_df, rows=2500, cols=50)
            append_status(sh, f"Parsed {len(odds_df)} odds events into {ODDS_TAB}")
    else:
        upsert_df(sh, ODDS_TAB, pd.DataFrame([{
            "message": "Parser not implemented for this provider yet. See FOOTBALL_ODDS_RAW."
        }]), rows=50, cols=5)
        append_status(sh, "Odds parser not implemented for provider; logged raw only")

    print("Step 6 complete.")


if __name__ == "__main__":
    main()
