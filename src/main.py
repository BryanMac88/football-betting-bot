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

TOP_N = 10
TOP_MIX = 20


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
    """
    Poisson model score matrix -> 1X2, BTTS, Over/Under 1.5 and 2.5
    """
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = 0.0
    p_btts = 0.0
    p_over15 = 0.0
    p_over25 = 0.0

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
            if tg > 1:
                p_over15 += p
            if tg > 2:
                p_over25 += p

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts,
        "p_btts_no": 1 - p_btts,
        "p_over_1_5": p_over15,
        "p_under_1_5": 1 - p_over15,
        "p_over_2_5": p_over25,
        "p_under_2_5": 1 - p_over25,
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


def set_visible_tabs(keep_titles: List[str]):
    """
    Hide all worksheets except keep_titles
    """
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


# ================= RANKING HELPERS =================
CORE_COLS = ["utcDate", "league", "home", "away"]


def top_n_for_market(
    probs_df: pd.DataFrame,
    prob_col: str,
    bet_label: str,
    top_n: int = TOP_N
) -> pd.DataFrame:
    """
    Returns a Top-N table for a single probability column.
    IMPORTANT: does NOT add 'rank' (we add rank only at final write stage).
    """
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    df = probs_df[CORE_COLS + [prob_col]].copy()
    df = df.rename(columns={prob_col: "prob"})
    df["bet"] = bet_label
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["prob"])
    df = df.sort_values(["prob", "utcDate"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "utcDate", "league", "home", "away", "bet", "prob"]]


def build_top20_mix(probs_df: pd.DataFrame, top_k: int = TOP_MIX) -> pd.DataFrame:
    """
    Combine all candidate bet types and return a single Top-K mixture.
    We REMOVE any existing rank columns before adding our own.
    """
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    parts = []

    # Build full candidate lists (not just Top10) then take Top20 overall
    def _all_for(col: str, label: str) -> pd.DataFrame:
        d = probs_df[CORE_COLS + [col]].copy()
        d = d.rename(columns={col: "prob"})
        d["bet"] = label
        d["prob"] = pd.to_numeric(d["prob"], errors="coerce")
        d = d.dropna(subset=["prob"])
        return d[["utcDate", "league", "home", "away", "bet", "prob"]]

    parts.append(_all_for("p_home", "HOME WIN"))
    parts.append(_all_for("p_away", "AWAY WIN"))
    parts.append(_all_for("p_draw", "DRAW"))
    parts.append(_all_for("p_btts_yes", "BTTS YES"))
    parts.append(_all_for("p_btts_no", "BTTS NO"))
    parts.append(_all_for("p_over_1_5", "OVER 1.5"))
    parts.append(_all_for("p_over_2_5", "OVER 2.5"))

    mix = pd.concat(parts, ignore_index=True)

    # Unique by fixture + bet
    mix = mix.drop_duplicates(subset=["utcDate", "home", "away", "bet"])

    # Sort & take top_k
    mix = mix.sort_values(["prob", "utcDate"], ascending=[False, True]).head(top_k).reset_index(drop=True)

    # Add rank once
    mix.insert(0, "rank", range(1, len(mix) + 1))

    return mix[["rank", "utcDate", "league", "home", "away", "bet", "prob"]]


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

    # Always write core tabs (even if empty)
    write_df("Fixtures", fx_df)

    keep_tabs = [
        "Fixtures", "Team_Form", "Picks", "Value_Bets",
        "Top10_Home_Win", "Top10_Away_Win", "Top10_Draw",
        "Top10_BTTS_Yes", "Top10_BTTS_No",
        "Top10_Over_1_5", "Top10_Over_2_5",
        "Top20_Mix",
    ]

    if rs_df.empty:
        write_df("Team_Form", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        write_df("Value_Bets", pd.DataFrame())

        write_df("Top10_Home_Win", pd.DataFrame())
        write_df("Top10_Away_Win", pd.DataFrame())
        write_df("Top10_Draw", pd.DataFrame())
        write_df("Top10_BTTS_Yes", pd.DataFrame())
        write_df("Top10_BTTS_No", pd.DataFrame())
        write_df("Top10_Over_1_5", pd.DataFrame())
        write_df("Top10_Over_2_5", pd.DataFrame())
        write_df("Top20_Mix", pd.DataFrame())

        set_visible_tabs(keep_tabs)
        log("=== DONE (no results) ===")
        return

    # ---------- TEAM FORM (safe join) ----------
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
    form = home.join(away, how="outer").fillna(0.0)

    league_home_gf = form["home_hg"].replace(0, pd.NA).mean()
    league_away_gf = form["away_hg"].replace(0, pd.NA).mean()
    league_home_gf = float(league_home_gf) if league_home_gf == league_home_gf else BASE_HOME
    league_away_gf = float(league_away_gf) if league_away_gf == league_away_gf else BASE_AWAY

    form["attack"] = (form["home_hg"].replace(0, league_home_gf) / league_home_gf).fillna(1.0)
    form["defense"] = (form["home_ag"].replace(0, league_away_gf) / league_away_gf).fillna(1.0)

    team_form_df = form.reset_index().rename(columns={"index": "team"})
    write_df("Team_Form", team_form_df)

    # ---------- Build probabilities for fixtures ----------
    if fx_df.empty:
        write_df("Picks", pd.DataFrame())
        write_df("Value_Bets", pd.DataFrame())

        write_df("Top10_Home_Win", pd.DataFrame())
        write_df("Top10_Away_Win", pd.DataFrame())
        write_df("Top10_Draw", pd.DataFrame())
        write_df("Top10_BTTS_Yes", pd.DataFrame())
        write_df("Top10_BTTS_No", pd.DataFrame())
        write_df("Top10_Over_1_5", pd.DataFrame())
        write_df("Top10_Over_2_5", pd.DataFrame())
        write_df("Top20_Mix", pd.DataFrame())

        set_visible_tabs(keep_tabs)
        log("=== DONE (no fixtures) ===")
        return

    probs_rows: List[Dict[str, Any]] = []
    form_idx = form  # indexed by team name

    for _, r in fx_df.iterrows():
        h = r.get("home")
        a = r.get("away")

        if h in form_idx.index and a in form_idx.index:
            lh = league_home_gf * float(form_idx.loc[h, "attack"]) * float(form_idx.loc[a, "defense"])
            la = league_away_gf
        else:
            lh, la = BASE_HOME, BASE_AWAY

        p = match_probs(lh, la)
        probs_rows.append({**r.to_dict(), **p})

    probs_df = pd.DataFrame(probs_rows)

    # ---------- Picks tab ----------
    def _pick_1x2(row: pd.Series) -> Tuple[str, float]:
        opts = [("HOME", row["p_home"]), ("DRAW", row["p_draw"]), ("AWAY", row["p_away"])]
        best = max(opts, key=lambda x: float(x[1]))
        return best[0], float(best[1])

    picks = []
    for _, row in probs_df.iterrows():
        pick, pbest = _pick_1x2(row)
        picks.append({
            "utcDate": row["utcDate"],
            "league": row["league"],
            "home": row["home"],
            "away": row["away"],
            "pick_1x2": pick,
            "p_1x2": round(pbest, 3),
            "p_home": round(float(row["p_home"]), 3),
            "p_draw": round(float(row["p_draw"]), 3),
            "p_away": round(float(row["p_away"]), 3),
            "p_btts_yes": round(float(row["p_btts_yes"]), 3),
            "p_btts_no": round(float(row["p_btts_no"]), 3),
            "p_over_1_5": round(float(row["p_over_1_5"]), 3),
            "p_under_1_5": round(float(row["p_under_1_5"]), 3),
            "p_over_2_5": round(float(row["p_over_2_5"]), 3),
            "p_under_2_5": round(float(row["p_under_2_5"]), 3),
        })

    picks_out = pd.DataFrame(picks).sort_values(["p_1x2", "utcDate"], ascending=[False, True])
    write_df("Picks", picks_out)

    # ---------- Value_Bets (still empty on free/no-odds) ----------
    write_df("Value_Bets", pd.DataFrame())

    # ---------- Top 10 per market ----------
    write_df("Top10_Home_Win", top_n_for_market(probs_df, "p_home", "HOME WIN", TOP_N))
    write_df("Top10_Away_Win", top_n_for_market(probs_df, "p_away", "AWAY WIN", TOP_N))
    write_df("Top10_Draw", top_n_for_market(probs_df, "p_draw", "DRAW", TOP_N))
    write_df("Top10_BTTS_Yes", top_n_for_market(probs_df, "p_btts_yes", "BTTS YES", TOP_N))
    write_df("Top10_BTTS_No", top_n_for_market(probs_df, "p_btts_no", "BTTS NO", TOP_N))
    write_df("Top10_Over_1_5", top_n_for_market(probs_df, "p_over_1_5", "OVER 1.5", TOP_N))
    write_df("Top10_Over_2_5", top_n_for_market(probs_df, "p_over_2_5", "OVER 2.5", TOP_N))

    # ---------- Top 20 mixture (FIXED) ----------
    mix_df = build_top20_mix(probs_df, TOP_MIX)
    write_df("Top20_Mix", mix_df)

    # ---------- CLEAN UP TABS ----------
    set_visible_tabs(keep_tabs)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
