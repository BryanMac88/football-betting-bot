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

# ---------------- CONFIG ----------------
COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",   # Top 5
    "CL", "EL", "EC",                 # Europe (may 403 on free)
    "ELC", "EL1", "EL2",              # England (EL1/EL2 often 403 on free)
    "SPL",                            # Scotland (often 403 on free)
    "SD",                             # Spain 2 (often 403 on free)
]

# football-data.org free tier: 10 req/min -> safe sleep
SLEEP_SECONDS = 7.0

DAYS_AHEAD = 7
HISTORY_DAYS = 210
MAX_GOALS = 10

# Baseline goal rates (fallback)
BASE_HOME = 1.45
BASE_AWAY = 1.20

# “Strong bet” thresholds (tune later)
THRESH_1X2 = 0.62
THRESH_BTTS = 0.60
THRESH_OU25 = 0.60
THRESH_OU15 = 0.66   # 1.5 is usually higher-prob, so use a higher threshold

TOP_N_BEST_BETS = 10

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
    p_over_25 = 0.0
    p_over_15 = 0.0

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

            tg = i + j
            if tg > 2:
                p_over_25 += p
            if tg > 1:
                p_over_15 += p

    return {
        "lambda_home": lh,
        "lambda_away": la,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts,
        "p_btts_no": 1 - p_btts,
        "p_over_2_5": p_over_25,
        "p_under_2_5": 1 - p_over_25,
        "p_over_1_5": p_over_15,
        "p_under_1_5": 1 - p_over_15,
    }

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

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
        r.raise_for_status()
        return r.json()

    def matches(self, code: str, status: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": date_from, "dateTo": date_to},
        ).get("matches", [])

# ---------------- BEST BETS ----------------
def build_best_bets(probs_df: pd.DataFrame) -> pd.DataFrame:
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for _, r in probs_df.iterrows():
        # 1X2 strongest
        p_home = safe_float(r.get("p_home"))
        p_draw = safe_float(r.get("p_draw"))
        p_away = safe_float(r.get("p_away"))
        best_1x2 = max([("HOME", p_home), ("DRAW", p_draw), ("AWAY", p_away)], key=lambda x: x[1])

        # BTTS
        p_btts_yes = safe_float(r.get("p_btts_yes"))
        p_btts_no = safe_float(r.get("p_btts_no"))
        best_btts = max([("BTTS_YES", p_btts_yes), ("BTTS_NO", p_btts_no)], key=lambda x: x[1])

        # O/U 2.5
        p_over_25 = safe_float(r.get("p_over_2_5"))
        p_under_25 = safe_float(r.get("p_under_2_5"))
        best_ou25 = max([("OVER_2_5", p_over_25), ("UNDER_2_5", p_under_25)], key=lambda x: x[1])

        # O/U 1.5
        p_over_15 = safe_float(r.get("p_over_1_5"))
        p_under_15 = safe_float(r.get("p_under_1_5"))
        best_ou15 = max([("OVER_1_5", p_over_15), ("UNDER_1_5", p_under_15)], key=lambda x: x[1])

        candidates = []
        if best_1x2[1] >= THRESH_1X2:
            candidates.append(("1X2", best_1x2[0], best_1x2[1]))
        if best_btts[1] >= THRESH_BTTS:
            candidates.append(("BTTS", best_btts[0], best_btts[1]))
        if best_ou25[1] >= THRESH_OU25:
            candidates.append(("O/U 2.5", best_ou25[0], best_ou25[1]))
        if best_ou15[1] >= THRESH_OU15:
            candidates.append(("O/U 1.5", best_ou15[0], best_ou15[1]))

        if candidates:
            market, pick, conf = max(candidates, key=lambda x: x[2])
            note = "meets threshold"
        else:
            # fallback: best overall probability
            all_best = [
                ("1X2", best_1x2[0], best_1x2[1]),
                ("BTTS", best_btts[0], best_btts[1]),
                ("O/U 2.5", best_ou25[0], best_ou25[1]),
                ("O/U 1.5", best_ou15[0], best_ou15[1]),
            ]
            market, pick, conf = max(all_best, key=lambda x: x[2])
            note = "below thresholds"

        rows.append({
            "utcDate": r.get("utcDate"),
            "comp": r.get("comp"),
            "home": r.get("home"),
            "away": r.get("away"),
            "recommended_market": market,
            "recommended_pick": pick,
            "confidence": round(conf, 4),
            "p_home": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away": round(p_away, 4),
            "p_btts_yes": round(p_btts_yes, 4),
            "p_over_2_5": round(p_over_25, 4),
            "p_over_1_5": round(p_over_15, 4),
            "note": note,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["confidence", "utcDate"], ascending=[False, True]).head(TOP_N_BEST_BETS).reset_index(drop=True)
    return out

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

    for code in COMP_CODES:
        if hit_429:
            break

        # SCHEDULED
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
                log(f"skip {code}: 403")
                continue
            if status == 429:
                hit_429 = True
                log(f"STOP: 429 hit on {code} SCHEDULED")
                break
            log(f"skip {code}: HTTP {status}")
            continue

        # FINISHED
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
                log(f"skip {code} results: 403")
                continue
            if status == 429:
                hit_429 = True
                log(f"STOP: 429 hit on {code} FINISHED")
                break
            log(f"skip {code} results: HTTP {status}")
            continue

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)

    write_df("Fixtures", fx_df)

    if fx_df.empty or rs_df.empty:
        write_df("Model_Probs", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        write_df("Best_Bets", pd.DataFrame())
        append_run_log({
            "ts": ts,
            "fixtures_rows": int(len(fx_df)),
            "results_rows": int(len(rs_df)),
            "blocked_403": ",".join(blocked_403),
            "hit_429": hit_429,
            "fixtures_calls": fixtures_calls,
            "results_calls": results_calls,
            "note": "No fixtures or no results available",
        })
        log("=== DONE (no model) ===")
        return

    # Team averages
    home_stats = (
        rs_df.groupby("home")[["hg", "ag"]]
        .mean()
        .rename(columns={"hg": "home_gf", "ag": "home_ga"})
    )
    away_stats = (
        rs_df.groupby("away")[["ag", "hg"]]
        .mean()
        .rename(columns={"ag": "away_gf", "hg": "away_ga"})
    )
    form = home_stats.join(away_stats, how="outer").fillna(0.0)

    league_home_gf = form["home_gf"].replace(0, pd.NA).mean()
    league_away_gf = form["away_gf"].replace(0, pd.NA).mean()
    league_home_gf = float(league_home_gf) if league_home_gf == league_home_gf else BASE_HOME
    league_away_gf = float(league_away_gf) if league_away_gf == league_away_gf else BASE_AWAY

    form["attack"] = (form["home_gf"].replace(0, league_home_gf) / league_home_gf).fillna(1.0)
    form["defense"] = (form["home_ga"].replace(0, league_away_gf) / league_away_gf).fillna(1.0)

    probs_rows: List[Dict[str, Any]] = []
    for _, r in fx_df.iterrows():
        h = r["home"]
        a = r["away"]
        if h in form.index and a in form.index:
            lh = league_home_gf * float(form.loc[h, "attack"]) * float(form.loc[a, "defense"])
            la = league_away_gf
        else:
            lh, la = league_home_gf, league_away_gf

        p = match_probs(lh, la)
        probs_rows.append({**r.to_dict(), **p})

    probs_df = pd.DataFrame(probs_rows)
    write_df("Model_Probs", probs_df)

    # Picks
    picks = []
    for _, r in probs_df.iterrows():
        pick_1x2 = max(
            [("HOME", r["p_home"]), ("DRAW", r["p_draw"]), ("AWAY", r["p_away"])],
            key=lambda x: x[1],
        )
        pick_btts = "BTTS_YES" if r["p_btts_yes"] >= 0.55 else ("BTTS_NO" if r["p_btts_no"] >= 0.60 else "")
        pick_ou25 = "OVER_2_5" if r["p_over_2_5"] >= 0.58 else ("UNDER_2_5" if r["p_under_2_5"] >= 0.62 else "")
        pick_ou15 = "OVER_1_5" if r["p_over_1_5"] >= 0.66 else ("UNDER_1_5" if r["p_under_1_5"] >= 0.72 else "")

        picks.append({
            "utcDate": r["utcDate"],
            "comp": r["comp"],
            "home": r["home"],
            "away": r["away"],
            "pick_1x2": pick_1x2[0],
            "p_1x2": round(float(pick_1x2[1]), 4),
            "pick_btts": pick_btts,
            "p_btts_yes": round(float(r["p_btts_yes"]), 4),
            "pick_ou25": pick_ou25,
            "p_over_2_5": round(float(r["p_over_2_5"]), 4),
            "pick_ou15": pick_ou15,
            "p_over_1_5": round(float(r["p_over_1_5"]), 4),
        })

    write_df("Picks", pd.DataFrame(picks))

    # Best Bets (Top 10)
    best_bets_df = build_best_bets(probs_df)
    write_df("Best_Bets", best_bets_df)

    append_run_log({
        "ts": ts,
        "fixtures_rows": int(len(fx_df)),
        "results_rows": int(len(rs_df)),
        "probs_rows": int(len(probs_df)),
        "picks_rows": int(len(picks)),
        "best_bets_rows": int(len(best_bets_df)),
        "blocked_403": ",".join(blocked_403),
        "hit_429": hit_429,
        "fixtures_calls": fixtures_calls,
        "results_calls": results_calls,
        "note": "OK",
    })

    log("=== DONE ===")

if __name__ == "__main__":
    main()
