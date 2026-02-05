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
def now() -> datetime:
    return datetime.now(timezone.utc)


def log(msg: str) -> None:
    print(msg, flush=True)


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def match_probs(lh: float, la: float) -> Dict[str, float]:
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = 0.0
    p_btts = 0.0
    p_over15 = 0.0

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
def gs() -> gspread.Client:
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
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(name, 2000, 40)

    ws.clear()
    if df is None or df.empty:
        ws.update([["(no data)"]])
    else:
        ws.update([list(df.columns)] + df.fillna("").values.tolist())


# ================= TAB VISIBILITY =================
def set_visible_tabs(keep_titles: List[str]):
    sh = open_sheet()
    meta = sh.fetch_sheet_metadata()

    reqs = []
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        title = props.get("title")
        sid = props.get("sheetId")
        if title is None or sid is None:
            continue

        hidden = title not in keep_titles
        reqs.append({
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "hidden": hidden},
                "fields": "hidden",
            }
        })

    if reqs:
        sh.batch_update({"requests": reqs})


# ================= FOOTBALL-DATA API =================
@dataclass
class FD:
    token: str
    base: str = "https://api.football-data.org/v4"

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.get(
            self.base + path,
            headers={"X-Auth-Token": self.token},
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def matches(self, code: str, status: str, d1: str, d2: str) -> List[Dict[str, Any]]:
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": d1, "dateTo": d2},
        ).get("matches", [])


# ================= MAIN =================
def main():
    log("=== START ===")
    fd = FD(env("FOOTBALL_DATA_TOKEN"))

    today = now().date()
    f1 = today.isoformat()
    f2 = (today + timedelta(days=DAYS_AHEAD)).isoformat()
    h1 = (today - timedelta(days=HISTORY_DAYS)).isoformat()

    fixtures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    # Fetch fixtures + results (best-effort)
    for c in COMP_CODES:
        # Fixtures
        time.sleep(SLEEP_SECONDS)
        try:
            fx = fd.matches(c, "SCHEDULED", f1, f2)
            log(f"{c} fixtures: {len(fx)}")
            for m in fx:
                fixtures.append({
                    "league": c,
                    "utcDate": m.get("utcDate"),
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                })
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            log(f"skip fixtures {c}: HTTP {status}")
        except Exception as e:
            log(f"skip fixtures {c}: {e}")

        # Results
        time.sleep(SLEEP_SECONDS)
        try:
            rs = fd.matches(c, "FINISHED", h1, f1)
            log(f"{c} results: {len(rs)}")
            for m in rs:
                sc = ((m.get("score") or {}).get("fullTime") or {})
                hg = sc.get("home")
                ag = sc.get("away")
                if hg is None or ag is None:
                    continue
                results.append({
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                    "hg": int(hg),
                    "ag": int(ag),
                })
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            log(f"skip results {c}: HTTP {status}")
        except Exception as e:
            log(f"skip results {c}: {e}")

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)

    # Always write the four tabs (even if empty)
    write_df("Fixtures", fx_df)

    # If no results, we can't build team form
    if rs_df.empty:
        write_df("Team_Form", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        write_df("Value_Bets", pd.DataFrame())
        set_visible_tabs(["Fixtures", "Team_Form", "Picks", "Value_Bets"])
        log("=== DONE (no results) ===")
        return

    # ---------- TEAM FORM (FIXED: no overlapping columns) ----------
    home = (
        rs_df.groupby("home")[["hg", "ag"]]
        .mean()
        .rename(columns={"hg": "home_hg", "ag": "home_ag"})
    )
    away = (
        rs_df.groupby("away")[["hg", "ag"]]
        .mean()
        .rename(columns={"hg": "away_hg", "ag": "away_ag"})
    )

    # Join now safe (no overlaps)
    form = home.join(away, how="outer").fillna(0.0)

    # Simple strength metrics
    # home_hg = avg goals scored at home, home_ag = avg conceded at home
    # away_ag = avg conceded away, away_hg = avg scored away (naming is just consistent here)
    league_home_gf = form["home_hg"].replace(0, pd.NA).mean()
    league_away_gf = form["away_hg"].replace(0, pd.NA).mean()

    league_home_gf = float(league_home_gf) if league_home_gf == league_home_gf else BASE_HOME
    league_away_gf = float(league_away_gf) if league_away_gf == league_away_gf else BASE_AWAY

    form["attack"] = (form["home_hg"].replace(0, league_home_gf) / league_home_gf).fillna(1.0)
    form["defense"] = (form["home_ag"].replace(0, league_away_gf) / league_away_gf).fillna(1.0)

    team_form_df = form.reset_index().rename(columns={"index": "team"})
    write_df("Team_Form", team_form_df)

    # ---------- PICKS ----------
    picks: List[Dict[str, Any]] = []
    if fx_df.empty:
        write_df("Picks", pd.DataFrame())
        write_df("Value_Bets", pd.DataFrame())
        set_visible_tabs(["Fixtures", "Team_Form", "Picks", "Value_Bets"])
        log("=== DONE (no fixtures) ===")
        return

    # Use team form if available, otherwise fallback to baseline
    form_idx = form  # index = team names

    for _, r in fx_df.iterrows():
        h = r.get("home")
        a = r.get("away")

        if h in form_idx.index and a in form_idx.index:
            lh = league_home_gf * float(form_idx.loc[h, "attack"]) * float(form_idx.loc[a, "defense"])
            la = league_away_gf
        else:
            lh, la = BASE_HOME, BASE_AWAY

        p = match_probs(lh, la)

        best = max(
            [("HOME", p["p_home"]), ("DRAW", p["p_draw"]), ("AWAY", p["p_away"])],
            key=lambda x: x[1],
        )

        picks.append({
            "utcDate": r.get("utcDate"),
            "home": h,
            "away": a,
            "pick_1x2": best[0],
            "p_1x2": round(float(best[1]), 3),
            "p_btts_yes": round(float(p["p_btts_yes"]), 3),
            "p_over_1_5": round(float(p["p_over_1_5"]), 3),
        })

    picks_df = pd.DataFrame(picks).sort_values("p_1x2", ascending=False)
    write_df("Picks", picks_df)

    # ---------- VALUE BETS ----------
    # We are not pulling bookmaker odds in this free setup, so keep it empty for now.
    # When you want, we can add odds again carefully (without hitting rate limits).
    write_df("Value_Bets", pd.DataFrame())

    # ---------- CLEAN UP TABS ----------
    set_visible_tabs(["Fixtures", "Team_Form", "Picks", "Value_Bets"])
    log("=== DONE ===")


if __name__ == "__main__":
    main()
