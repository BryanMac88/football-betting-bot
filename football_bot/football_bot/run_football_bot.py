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

COMPETITIONS = ["PL"]   # Premier League (safe for free tier)
LOOKBACK_DAYS = 365

BASE_URL = "https://api.football-data.org/v4"

# =========================================


def get_headers():
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Missing FOOTBALL_DATA_TOKEN")
    return {"X-Auth-Token": token}


def fetch_matches(comp, date_from, date_to):
    url = f"{BASE_URL}/competitions/{comp}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to}

    r = requests.get(url, headers=get_headers(), params=params, timeout=30)
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


def write_sheet(df, sheet, tab_name):
    try:
        ws = sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=tab_name, rows=2000, cols=20)

    if df.empty:
        return

    values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
    ws.clear()
    ws.update(values)


def append_status(sheet, message):
    try:
        ws = sheet.worksheet(STATUS_TAB)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=STATUS_TAB, rows=200, cols=10)
        ws.append_row(["timestamp_utc", "message"])

    ws.append_row([datetime.now(timezone.utc).isoformat(), message])


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

    today = datetime.now(timezone.utc).date()
    date_from = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    date_to = (today + timedelta(days=7)).isoformat()

    all_matches = []

    for comp in COMPETITIONS:
        df = fetch_matches(comp, date_from, date_to)
        all_matches.append(df)

    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    write_sheet(matches_df, sh, MATCHES_TAB)

    append_status(sh, f"Loaded {len(matches_df)} matches from football-data")

    print(f"Loaded {len(matches_df)} matches")


if __name__ == "__main__":
    main()
