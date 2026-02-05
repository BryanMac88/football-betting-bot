from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials


# ================= CONFIG =================
COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",
    "CL", "EL", "EC",
    "ELC", "EL1", "EL2",
    "SPL", "SD",
]

SLEEP_SECONDS = 7.0
DAYS_AHEAD = 7
HISTORY_DAYS = 210
MAX_GOALS = 10

BASE_HOME = 1.45
BASE_AWAY = 1.20


# ================= UTILS =================
def now():
    return datetime.now(timezone.utc)


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def poisson(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def match_probs(lh, la):
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = p_btts = p_over15 = 0.0

    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]

            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

            if i > 0 and j > 0:
                p_btts += p

            if i + j > 1:
                p_over15 += p

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts,
        "p_over_1_5": p_over15,
    }


# ================= GOOGLE SHEETS =================
def gs():
    creds = Credentials.from_service_account_info(
        json.loads(env("GOOGLE_SERVICE_ACCOUNT_JSON")),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)


def open_sheet():
    return gs().open_by_key(env("SHEET_ID"))


def write_df(name: str, df: pd.DataFrame):
    sh = open_sheet()
    try:
        ws = sh.worksheet(name)
    except:
        ws = sh.add_worksheet(name, 2000, 40)

    ws.clear()
    if df.empty:
        ws.update([["(no data)"]])
    else:
        ws.update([list(df.columns)] + df.fillna("").values.tolist())


# ================= TAB VISIBILITY =================
def set_visible_tabs(keep_titles: List[str]):
    sh = open_sheet()
    meta = sh.fetch_sheet_metadata()

    reqs = []
    for s in meta["sheets"]:
        props = s["properties"]
        title = props["title"]
        sid = props["sheetId"]

        hidden = title not in keep_titles
        reqs.append({
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "hidden": hidden},
                "fields": "hidden",
            }
        })

    sh.batch_update({"requests": reqs})


# ================= FOOTBALL DATA =================
@dataclass
class FD:
    token: str
    base: str = "https://api.football-data.org/v4"

    def get(self, path, params):
        r = requests.get(
            self.base + path,
            headers={"X-Auth-Token": self.token},
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def matches(self, code, status, d1, d2):
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": d1, "dateTo": d2},
        ).get("matches", [])


# ================= MAIN =================
def main():
    fd = FD(env("FOOTBALL_DATA_TOKEN"))

    today = now().date()
    f1 = today.isoformat()
    f2 = (today + timedelta(days=DAYS_AHEAD)).isoformat()
    h1 = (today - timedelta(days=HISTORY_DAYS)).isoformat()

    fixtures = []
    results = []

    for c in COMP_CODES:
        time.sleep(SLEEP_SECONDS)
        try:
            fx = fd.matches(c, "SCHEDULED", f1, f2)
            for m in fx:
                fixtures.append({
                    "league": c,
                    "utcDate": m["utcDate"],
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                })
        except:
            continue

        time.sleep(SLEEP_SECONDS)
        try:
            rs = fd.matches(c, "FINISHED", h1, f1)
            for m in rs:
                sc = m["score"]["fullTime"]
                if sc["home"] is not None:
                    results.append({
                        "home": m["homeTeam"]["name"],
                        "away": m["awayTeam"]["name"],
                        "hg": sc["home"],
                        "ag": sc["away"],
                    })
        except:
            continue

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)

    write_df("Fixtures", fx_df)

    if rs_df.empty:
        write_df("Team_Form", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        write_df("Value_Bets", pd.DataFrame())
        return

    # ---------- TEAM FORM ----------
    home = rs_df.groupby("home")[["hg", "ag"]].mean()
    away = rs_df.groupby("away")[["ag", "hg"]].mean()
    form = home.join(away, how="outer").fillna(0)

    league_avg = form["hg"].mean() if not form.empty else BASE_HOME
    form["attack"] = form["hg"] / league_avg
    form["defense"] = form["ag"] / league_avg

    form.reset_index(inplace=True)
    form.rename(columns={"index": "team"}, inplace=True)
    write_df("Team_Form", form)

    # ---------- MODEL / PICKS ----------
    picks = []

    for _, r in fx_df.iterrows():
        lh, la = BASE_HOME, BASE_AWAY
        p = match_probs(lh, la)

        best = max(
            [("HOME", p["p_home"]), ("DRAW", p["p_draw"]), ("AWAY", p["p_away"])],
            key=lambda x: x[1],
        )

        picks.append({
            "utcDate": r["utcDate"],
            "home": r["home"],
            "away": r["away"],
            "pick": best[0],
            "probability": round(best[1], 3),
            "btts_yes": round(p["p_btts_yes"], 3),
            "over_1_5": round(p["p_over_1_5"], 3),
        })

    picks_df = pd.DataFrame(picks).sort_values("probability", ascending=False)
    write_df("Picks", picks_df)

    # ---------- VALUE BETS (placeholder for now) ----------
    write_df("Value_Bets", pd.DataFrame())

    # ---------- CLEAN UP TABS ----------
    set_visible_tabs(["Fixtures", "Team_Form", "Picks", "Value_Bets"])


if __name__ == "__main__":
    main()
