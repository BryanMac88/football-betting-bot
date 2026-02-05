from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials

# ---------------- CONFIG ----------------
# Keep your full wish-list here; we will auto-skip what your plan can't access.
COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",   # Top 5
    "CL", "EL", "EC",                 # Europe
    "ELC", "EL1", "EL2",              # England
    "SPL",                            # Scotland
    "SD",                             # Spain 2
]

# football-data.org free tier: 10 req/min -> sleep to be safe
SLEEP_SECONDS = 7.0

DAYS_AHEAD = 7
HISTORY_DAYS = 210
MAX_GOALS = 10

# Baseline league goal rates (fallback)
BASE_HOME = 1.45
BASE_AWAY = 1.20

# ---------------- UTILS ----------------
def now() -> datetime:
    return datetime.now(timezone.utc)

def iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")

def log(msg: str) -> None:
    print(msg, flush=True)

def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def match_probs(lh: float, la: float) -> Dict[str, float]:
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = 0.0
    p_btts = 0.0
    p_over = 0.0

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
            if i + j > 2:
                p_over += p

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts,
        "p_btts_no": 1 - p_btts,
        "p_over_2_5": p_over,
        "p_under_2_5": 1 - p_over,
        "lambda_home": lh,
        "lambda_away": la,
    }

# ---------------- GOOGLE SHEETS ----------------
def gs_client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        json.loads(env("GOOGLE_SERVICE_ACCOUNT_JSON")),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)

def write_df(tab: str, df: pd.DataFrame) -> None:
    sh = gs_client().open_by_key(env("SHEET_ID"))
    try:
        ws = sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=2000, cols=40)

    ws.clear()
    if df is None or df.empty:
        ws.update([["(no data)"]])
    else:
        ws.update([list(df.columns)] + df.fillna("").values.tolist())

def append_run_log(row: Dict[str, Any]) -> None:
    sh = gs_client().open_by_key(env("SHEET_ID"))
    try:
        ws = sh.worksheet("Run_Log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Run_Log", rows=2000, cols=40)
        ws.append_row(list(row.keys()))

    header = ws.row_values(1)
    if not header:
        ws.append_row(list(row.keys()))
        header = list(row.keys())

    # expand header if new keys appear
    missing = [k for k in row.keys() if k not in header]
    if missing:
        header.extend(missing)
        ws.clear()
        ws.append_row(header)

    ws.append_row([row.get(k, "") for k in header])

# ---------------- FOOTBALL-DATA API ----------------
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
        # Raise with context
        r.raise_for_status()
        return r.json()

    def matches(self, code: str, status: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": date_from, "dateTo": date_to},
        ).get("matches", [])

# ---------------- MAIN ----------------
def main() -> None:
    ts = iso(now())
    log(f"=== RUN {ts} ===")

    fd = FD(env("FOOTBALL_DATA_TOKEN"))

    today = now().date()
    fixtures_from = today.isoformat()
    fixtures_to = (today + timedelta(days=DAYS_AHEAD)).isoformat()
    hist_from = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    hist_to = today.isoformat()

    fixtures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    blocked_403: List[str] = []
    hit_429 = False

    fixtures_calls = 0
    results_calls = 0

    # Fetch data
    for code in COMP_CODES:
        # stop entirely if we hit rate limit
        if hit_429:
            break

        # ---- SCHEDULED fixtures ----
        time.sleep(SLEEP_SECONDS)
        try:
            ms = fd.matches(code, "SCHEDULED", fixtures_from, fixtures_to)
            fixtures_calls += 1
            for m in ms:
                fixtures.append({
                    "comp": code,
                    "match_id": m.get("id"),
                    "utcDate": m.get("utcDate"),
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                })
            log(f"{code} SCHEDULED: {len(ms)}")
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 403:
                blocked_403.append(code)
                log(f"skip {code}: 403 (not available on your plan)")
                continue
            if status == 429:
                hit_429 = True
                log(f"STOP: 429 rate limit hit on {code} SCHEDULED")
                break
            log(f"skip {code}: HTTP {status}")
            continue
        except Exception as e:
            log(f"skip {code}: {e}")
            continue

        # ---- FINISHED results ----
        time.sleep(SLEEP_SECONDS)
        try:
            ms = fd.matches(code, "FINISHED", hist_from, hist_to)
            results_calls += 1
            for m in ms:
                sc = ((m.get("score") or {}).get("fullTime") or {})
                hg = sc.get("home")
                ag = sc.get("away")
                if hg is None or ag is None:
                    continue
                results.append({
                    "comp": code,
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                    "hg": int(hg),
                    "ag": int(ag),
                })
            log(f"{code} FINISHED: {len(ms)}")
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 403:
                if code not in blocked_403:
                    blocked_403.append(code)
                log(f"skip {code} results: 403 (not available on your plan)")
                continue
            if status == 429:
                hit_429 = True
                log(f"STOP: 429 rate limit hit on {code} FINISHED")
                break
            log(f"skip {code} results: HTTP {status}")
            continue
        except Exception as e:
            log(f"skip {code} results: {e}")
            continue

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)

    # Always write fixtures
    write_df("Fixtures", fx_df)

    # If we have no fixtures or no results, write blanks and log
    if fx_df.empty or rs_df.empty:
        write_df("Model_Probs", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        append_run_log({
            "ts": ts,
            "fixtures_rows": int(len(fx_df)),
            "results_rows": int(len(rs_df)),
            "blocked_403": ",".join(blocked_403),
            "hit_429": hit_429,
            "fixtures_calls": fixtures_calls,
            "results_calls": results_calls,
            "note": "No fixtures or no results available (check 403/429 coverage & limits)",
        })
        log("=== DONE (no model) ===")
        return

    # ---- Build simple team averages ----
    # Home averages
    home_stats = (
        rs_df.groupby("home")[["hg", "ag"]]
        .mean()
        .rename(columns={"hg": "home_gf", "ag": "home_ga"})
    )
    # Away averages
    away_stats = (
        rs_df.groupby("away")[["ag", "hg"]]
        .mean()
        .rename(columns={"ag": "away_gf", "hg": "away_ga"})
    )
    # Join without overlap
    form = home_stats.join(away_stats, how="outer").fillna(0.0)

    # Convert to attack/defense multipliers
    # League baseline derived from totals
    league_home_gf = form["home_gf"].replace(0, pd.NA).mean()
    league_away_gf = form["away_gf"].replace(0, pd.NA).mean()
    league_home_gf = float(league_home_gf) if league_home_gf == league_home_gf else BASE_HOME
    league_away_gf = float(league_away_gf) if league_away_gf == league_away_gf else BASE_AWAY

    form["attack"] = (form["home_gf"].replace(0, league_home_gf) / league_home_gf).fillna(1.0)
    form["defense"] = (form["home_ga"].replace(0, league_away_gf) / league_away_gf).fillna(1.0)

    # ---- Build probabilities for fixtures ----
    probs_rows: List[Dict[str, Any]] = []
    form_index = form

    for _, r in fx_df.iterrows():
        h = r["home"]
        a = r["away"]

        # fallback if team missing
        if h in form_index.index and a in form_index.index:
            lh = league_home_gf * float(form_index.loc[h, "attack"]) * float(form_index.loc[a, "defense"])
            la = league_away_gf
        else:
            lh, la = league_home_gf, league_away_gf

        p = match_probs(lh, la)
        probs_rows.append({**r.to_dict(), **p})

    probs_df = pd.DataFrame(probs_rows)
    write_df("Model_Probs", probs_df)

    # ---- Picks ----
    picks = []
    for _, r in probs_df.iterrows():
        pick_1x2 = max(
            [("HOME", r["p_home"]), ("DRAW", r["p_draw"]), ("AWAY", r["p_away"])],
            key=lambda x: x[1],
        )
        pick_btts = "BTTS_YES" if r["p_btts_yes"] >= 0.55 else ("BTTS_NO" if r["p_btts_no"] >= 0.60 else "")
        pick_totals = "OVER_2_5" if r["p_over_2_5"] >= 0.58 else ("UNDER_2_5" if r["p_under_2_5"] >= 0.62 else "")

        picks.append({
            "utcDate": r["utcDate"],
            "comp": r["comp"],
            "home": r["home"],
            "away": r["away"],
            "pick_1x2": pick_1x2[0],
            "p_1x2": round(float(pick_1x2[1]), 4),
            "pick_btts": pick_btts,
            "p_btts_yes": round(float(r["p_btts_yes"]), 4),
            "pick_totals": pick_totals,
            "p_over_2_5": round(float(r["p_over_2_5"]), 4),
        })

    picks_df = pd.DataFrame(picks)
    write_df("Picks", picks_df)

    append_run_log({
        "ts": ts,
        "fixtures_rows": int(len(fx_df)),
        "results_rows": int(len(rs_df)),
        "probs_rows": int(len(probs_df)),
        "picks_rows": int(len(picks_df)),
        "blocked_403": ",".join(blocked_403),
        "hit_429": hit_429,
        "fixtures_calls": fixtures_calls,
        "results_calls": results_calls,
        "note": "OK" if not hit_429 else "Hit 429 (rate limit) — reduce competitions or run less frequently",
    })

    log("=== DONE ===")


if __name__ == "__main__":
    main()
