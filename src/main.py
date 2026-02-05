from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials

# ---------------- CONFIG ----------------
COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",   # Top 5
    "CL", "EL", "EC",                 # Europe
    "ELC", "EL1", "EL2",              # England
    "SPL",                            # Scotland
    "SD",                             # Spain 2
]

SLEEP = 7.0
DAYS_AHEAD = 7
HISTORY_DAYS = 210
MAX_GOALS = 10

# ---------------- UTILS ----------------
def now() -> datetime:
    return datetime.now(timezone.utc)

def log(msg: str) -> None:
    print(msg, flush=True)

def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def probs(lh: float, la: float) -> Dict[str, float]:
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]
    p_home = p_draw = p_away = p_btts = p_over = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i > 0 and j > 0: p_btts += p
            if i + j > 2: p_over += p
    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts,
        "p_btts_no": 1 - p_btts,
        "p_over_2_5": p_over,
        "p_under_2_5": 1 - p_over,
    }

# ---------------- GOOGLE SHEETS ----------------
def gs_client():
    creds = Credentials.from_service_account_info(
        json.loads(env("GOOGLE_SERVICE_ACCOUNT_JSON")),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)

def write_df(name: str, df: pd.DataFrame):
    sh = gs_client().open_by_key(env("SHEET_ID"))
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(name, 2000, 40)
    ws.clear()
    if df.empty:
        ws.update([["(no data)"]])
    else:
        ws.update([list(df.columns)] + df.fillna("").values.tolist())

def log_run(row: Dict):
    sh = gs_client().open_by_key(env("SHEET_ID"))
    try:
        ws = sh.worksheet("Run_Log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet("Run_Log", 2000, 20)
        ws.append_row(list(row.keys()))
    ws.append_row(list(row.values()))

# ---------------- FOOTBALL-DATA API ----------------
@dataclass
class FD:
    token: str
    base: str = "https://api.football-data.org/v4"

    def get(self, path: str, params=None):
        r = requests.get(
            self.base + path,
            headers={"X-Auth-Token": self.token},
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def matches(self, code, status, start, end):
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": start, "dateTo": end},
        ).get("matches", [])

# ---------------- MAIN ----------------
def main():
    log("=== START ===")
    fd = FD(env("FOOTBALL_DATA_TOKEN"))

    today = now().date()
    from_d = today.isoformat()
    to_d = (today + timedelta(days=DAYS_AHEAD)).isoformat()
    hist_from = (today - timedelta(days=HISTORY_DAYS)).isoformat()

    fixtures = []
    results = []

    for c in COMP_CODES:
        time.sleep(SLEEP)
        try:
            fx = fd.matches(c, "SCHEDULED", from_d, to_d)
            for m in fx:
                fixtures.append({
                    "comp": c,
                    "utcDate": m["utcDate"],
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                })
            rs = fd.matches(c, "FINISHED", hist_from, from_d)
            for m in rs:
                sc = m["score"]["fullTime"]
                if sc["home"] is not None:
                    results.append({
                        "home": m["homeTeam"]["name"],
                        "away": m["awayTeam"]["name"],
                        "hg": sc["home"],
                        "ag": sc["away"],
                    })
        except Exception as e:
            log(f"skip {c}: {e}")

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)

    write_df("Fixtures", fx_df)

    if rs_df.empty or fx_df.empty:
        write_df("Model_Probs", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        log_run({"ts": now().isoformat(), "note": "no data"})
        return

    home = rs_df.groupby("home")[["hg", "ag"]].mean()
    away = rs_df.groupby("away")[["ag", "hg"]].mean()
    form = home.join(away, how="outer").fillna(1.0)

    rows = []
    for _, r in fx_df.iterrows():
        lh = 1.45
        la = 1.20
        p = probs(lh, la)
        rows.append({**r, **p})

    probs_df = pd.DataFrame(rows)
    write_df("Model_Probs", probs_df)

    picks = []
    for _, r in probs_df.iterrows():
        pick = max(
            [("HOME", r["p_home"]), ("DRAW", r["p_draw"]), ("AWAY", r["p_away"])],
            key=lambda x: x[1],
        )
        picks.append({
            "utcDate": r["utcDate"],
            "home": r["home"],
            "away": r["away"],
            "pick": pick[0],
            "prob": round(pick[1], 4),
        })

    write_df("Picks", pd.DataFrame(picks))
    log_run({"ts": now().isoformat(), "fixtures": len(fx_df), "picks": len(picks)})
    log("=== DONE ===")

if __name__ == "__main__":
    main()
