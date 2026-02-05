from __future__ import annotations
import json
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _client_from_service_account_json(sa_json: str) -> gspread.Client:
    info = json.loads(sa_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def _ensure_worksheet(sh, title: str, rows: int = 1000, cols: int = 30):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def write_df(sh, title: str, df: pd.DataFrame):
    ws = _ensure_worksheet(sh, title, rows=max(1000, len(df)+10), cols=max(30, len(df.columns)+5))
    ws.clear()
    if df.empty:
        ws.update("A1", [["(no data)"]])
        return
    values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values)

def append_log(sh, message: str):
    ws = _ensure_worksheet(sh, "Run_Log", rows=200, cols=5)
    existing = ws.get_all_values()
    if not existing:
        ws.update("A1", [["utc_time","message"]])
        existing = ws.get_all_values()
    from datetime import datetime, timezone
    ws.append_row([datetime.now(timezone.utc).replace(microsecond=0).isoformat(), message])
