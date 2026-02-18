import os
import json
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

TAB_NAME = "FOOTBALL_STATUS"


def main() -> None:
    sheet_id = os.getenv("SHEET_ID")
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not sheet_id:
        raise RuntimeError("Missing SHEET_ID env var")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var")

    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # Create the status tab if it doesn't exist
    try:
        ws = sh.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB_NAME, rows=200, cols=10)
        ws.append_row(["timestamp_utc", "message"])

    ws.append_row([datetime.now(timezone.utc).isoformat(), "Football workflow ran OK"])
    print(f"Wrote heartbeat row to Google Sheet tab: {TAB_NAME}")


if __name__ == "__main__":
    main()
