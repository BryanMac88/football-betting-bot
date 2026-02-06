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
BASE_COMP_CODES = [
    "PL", "PD", "SA", "BL1", "FL1",
    "CL", "EL", "EC",
    "ELC", "EL1", "EL2",
    "SPL", "SD",
]

OPTIONAL_COMP_CODES = [
    "DED",  # Netherlands Eredivisie
    "PPL",  # Portugal Primeira Liga
    "BSA",  # Brazil Serie A
]

SLEEP_SECONDS = 7.0

DAYS_AHEAD = 3
HISTORY_DAYS = 210
MAX_GOALS = 10

TOP_N = 10
TOP_MIX = 20

UNIQUE_TEAMS_PER_TOP10 = True
UNIQUE_TEAMS_IN_TOP20 = True

# Recency weighting
RECENT_N = 10
RECENT_WEIGHT = 1.8

# Bayesian shrinkage
SHRINK_K = 8.0

# Head-to-head adjustment (optional)
USE_H2H = True
H2H_MATCHES_LOOKBACK = 6
H2H_SHRINK_K = 4.0
H2H_MAX_GOAL_ADJ = 0.15
H2H_BLEND = 0.35

# Confidence scoring
# Higher = more strict about needing sample size
CONF_K = 12.0   # "virtual matches" for confidence curve


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


def safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def match_probs(lh: float, la: float) -> Dict[str, float]:
    ph = [poisson(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [poisson(la, j) for j in range(MAX_GOALS + 1)]

    p_home = p_draw = p_away = 0.0

    p_btts_yes = 0.0
    p_over_0_5 = 0.0
    p_over_1_5 = 0.0
    p_over_2_5 = 0.0
    p_over_3_5 = 0.0

    p_home_over_0_5 = 0.0
    p_away_over_0_5 = 0.0

    p_home_cs = 0.0
    p_away_cs = 0.0

    p_home_win_to_nil = 0.0
    p_away_win_to_nil = 0.0

    p_btts_yes_over_2_5 = 0.0
    p_btts_no_under_2_5 = 0.0

    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = ph[i] * pa[j]

            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

            btts = (i > 0 and j > 0)
            if btts:
                p_btts_yes += p

            tg = i + j
            if tg > 0:
                p_over_0_5 += p
            if tg > 1:
                p_over_1_5 += p
            if tg > 2:
                p_over_2_5 += p
            if tg > 3:
                p_over_3_5 += p

            if i > 0:
                p_home_over_0_5 += p
            if j > 0:
                p_away_over_0_5 += p

            if j == 0:
                p_home_cs += p
            if i == 0:
                p_away_cs += p

            if i > j and j == 0:
                p_home_win_to_nil += p
            if j > i and i == 0:
                p_away_win_to_nil += p

            if btts and tg > 2:
                p_btts_yes_over_2_5 += p
            if (not btts) and tg <= 2:
                p_btts_no_under_2_5 += p

    p_btts_no = 1.0 - p_btts_yes

    p_1x = p_home + p_draw
    p_x2 = p_draw + p_away
    p_12 = p_home + p_away

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_1x": p_1x,
        "p_x2": p_x2,
        "p_12": p_12,
        "p_home_dnb_win": p_home,
        "p_away_dnb_win": p_away,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
        "p_over_0_5": p_over_0_5,
        "p_over_1_5": p_over_1_5,
        "p_over_2_5": p_over_2_5,
        "p_over_3_5": p_over_3_5,
        "p_under_0_5": 1.0 - p_over_0_5,
        "p_under_1_5": 1.0 - p_over_1_5,
        "p_under_2_5": 1.0 - p_over_2_5,
        "p_under_3_5": 1.0 - p_over_3_5,
        "p_home_over_0_5": p_home_over_0_5,
        "p_away_over_0_5": p_away_over_0_5,
        "p_home_cs": p_home_cs,
        "p_away_cs": p_away_cs,
        "p_home_win_to_nil": p_home_win_to_nil,
        "p_away_win_to_nil": p_away_win_to_nil,
        "p_btts_yes_over_2_5": p_btts_yes_over_2_5,
        "p_btts_no_under_2_5": p_btts_no_under_2_5,
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
        ws = sh.add_worksheet(name, 4000, 60)

    ws.clear()
    if df is None or df.empty:
        ws.update([["(no data)"]])
    else:
        ws.update([list(df.columns)] + df.fillna("").values.tolist())


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

    def get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        r = requests.get(
            self.base + path,
            headers={"X-Auth-Token": self.token},
            params=params or {},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def matches(self, code: str, status: str, d1: str, d2: str) -> List[Dict[str, Any]]:
        return self.get(
            f"/competitions/{code}/matches",
            {"status": status, "dateFrom": d1, "dateTo": d2},
        ).get("matches", [])

    def competitions(self) -> List[Dict[str, Any]]:
        return self.get("/competitions", {}).get("competitions", [])


# ================= FREE MODEL UPGRADES =================
def recency_weighted_mean(values: List[float], recent_n: int = RECENT_N, recent_weight: float = RECENT_WEIGHT) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    n = len(vals)
    weights = [1.0] * n
    for i in range(max(0, n - recent_n), n):
        weights[i] = recent_weight
    wsum = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / wsum


def shrink(mean_est: float, n: float, prior_mean: float, k: float = SHRINK_K) -> float:
    if n <= 0 or math.isnan(mean_est):
        return prior_mean
    return (n * mean_est + k * prior_mean) / (n + k)


def compute_league_baselines(rs: pd.DataFrame) -> pd.DataFrame:
    g = rs.groupby("league")
    out = pd.DataFrame({
        "home_gf": g["hg"].mean(),
        "away_gf": g["ag"].mean(),
        "n": g.size(),
    })
    return out


def compute_team_indices(rs: pd.DataFrame, league_baselines: pd.DataFrame) -> pd.DataFrame:
    rs = rs.sort_values("utcDate")
    rows = []

    for league, sub in rs.groupby("league"):
        base_home_gf = safe_float(league_baselines.loc[league, "home_gf"], 1.35) if league in league_baselines.index else 1.35
        base_away_gf = safe_float(league_baselines.loc[league, "away_gf"], 1.10) if league in league_baselines.index else 1.10

        # HOME venue per team
        for team, tsub in sub.groupby("home"):
            hg_list = tsub["hg"].tolist()
            ag_list = tsub["ag"].tolist()
            n = len(hg_list)

            home_hg = recency_weighted_mean(hg_list)
            home_ag = recency_weighted_mean(ag_list)

            home_hg_s = shrink(home_hg, n, base_home_gf)
            home_ag_s = shrink(home_ag, n, base_away_gf)

            rows.append({
                "league": league, "team": team,
                "home_hg": home_hg_s, "home_ag": home_ag_s, "n_home": n
            })

        # AWAY venue per team
        for team, tsub in sub.groupby("away"):
            ag_list = tsub["ag"].tolist()  # away scored
            hg_list = tsub["hg"].tolist()  # away conceded
            n = len(ag_list)

            away_hg = recency_weighted_mean(ag_list)
            away_ag = recency_weighted_mean(hg_list)

            away_hg_s = shrink(away_hg, n, base_away_gf)
            away_ag_s = shrink(away_ag, n, base_home_gf)

            rows.append({
                "league": league, "team": team,
                "away_hg": away_hg_s, "away_ag": away_ag_s, "n_away": n
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["league", "team"], as_index=False).agg({
        "home_hg": "max",
        "home_ag": "max",
        "away_hg": "max",
        "away_ag": "max",
        "n_home": "max",
        "n_away": "max",
    }).fillna(0.0)

    def _idx(row: pd.Series) -> pd.Series:
        league = row["league"]
        base_home_gf = safe_float(league_baselines.loc[league, "home_gf"], 1.35) if league in league_baselines.index else 1.35
        base_away_gf = safe_float(league_baselines.loc[league, "away_gf"], 1.10) if league in league_baselines.index else 1.10

        home_attack = (row["home_hg"] / base_home_gf) if base_home_gf > 0 else 1.0
        home_def = (row["home_ag"] / base_away_gf) if base_away_gf > 0 else 1.0
        away_attack = (row["away_hg"] / base_away_gf) if base_away_gf > 0 else 1.0
        away_def = (row["away_ag"] / base_home_gf) if base_home_gf > 0 else 1.0

        # clamp
        return pd.Series({
            "home_attack": clamp(float(home_attack), 0.55, 1.75),
            "home_defense": clamp(float(home_def), 0.55, 1.75),
            "away_attack": clamp(float(away_attack), 0.55, 1.75),
            "away_defense": clamp(float(away_def), 0.55, 1.75),
        })

    idxs = agg.apply(_idx, axis=1)
    out = pd.concat([agg, idxs], axis=1)
    out["n_total"] = out["n_home"].fillna(0.0) + out["n_away"].fillna(0.0)
    out = out.set_index(["league", "team"])
    return out


def h2h_goal_adjustment(rs: pd.DataFrame, league: str, home: str, away: str) -> Tuple[float, float]:
    if not USE_H2H or rs.empty:
        return 0.0, 0.0

    sub = rs[(rs["league"] == league) & (
        ((rs["home"] == home) & (rs["away"] == away)) |
        ((rs["home"] == away) & (rs["away"] == home))
    )].sort_values("utcDate")

    if sub.empty:
        return 0.0, 0.0

    sub = sub.tail(H2H_MATCHES_LOOKBACK)

    hg = []
    ag = []
    for _, r in sub.iterrows():
        if r["home"] == home and r["away"] == away:
            hg.append(float(r["hg"]))
            ag.append(float(r["ag"]))
        else:
            hg.append(float(r["ag"]))
            ag.append(float(r["hg"]))

    n = len(hg)
    if n == 0:
        return 0.0, 0.0

    h2h_hg = recency_weighted_mean(hg, recent_n=min(RECENT_N, n), recent_weight=RECENT_WEIGHT)
    h2h_ag = recency_weighted_mean(ag, recent_n=min(RECENT_N, n), recent_weight=RECENT_WEIGHT)

    mean_total = (h2h_hg + h2h_ag) / 2.0
    dh = (h2h_hg - mean_total)
    da = (h2h_ag - mean_total)

    shrink_factor = n / (n + H2H_SHRINK_K)
    dh *= shrink_factor * H2H_BLEND
    da *= shrink_factor * H2H_BLEND

    dh = clamp(dh, -H2H_MAX_GOAL_ADJ, H2H_MAX_GOAL_ADJ)
    da = clamp(da, -H2H_MAX_GOAL_ADJ, H2H_MAX_GOAL_ADJ)

    return float(dh), float(da)


def confidence_score(n_home: float, n_away: float) -> float:
    """
    0..1 confidence based on match counts.
    """
    n = max(0.0, float(n_home or 0.0) + float(n_away or 0.0))
    return float(n / (n + CONF_K))


# ================= RANKING HELPERS =================
CORE_COLS = ["utcDate", "league", "home", "away"]


def _dedupe_teams(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    used = set()
    kept = []
    for _, r in df.iterrows():
        h = r.get("home")
        a = r.get("away")
        if h in used or a in used:
            continue
        kept.append(r)
        if h:
            used.add(h)
        if a:
            used.add(a)
        if len(kept) >= limit:
            break
    if not kept:
        return df.head(0)
    return pd.DataFrame(kept).reset_index(drop=True)


def top_n_for_market(
    probs_df: pd.DataFrame,
    prob_col: str,
    bet_label: str,
    top_n: int = TOP_N,
    unique_teams: bool = UNIQUE_TEAMS_PER_TOP10,
) -> pd.DataFrame:
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    cols = CORE_COLS + ["confidence", prob_col]
    df = probs_df[cols].copy()
    df = df.rename(columns={prob_col: "prob"})
    df["bet"] = bet_label

    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["prob"])

    # Score: prob weighted by confidence to avoid tiny-sample teams floating to top
    df["score"] = df["prob"] * df["confidence"].fillna(0.5)

    df = df.sort_values(["score", "prob", "utcDate"], ascending=[False, False, True]).reset_index(drop=True)

    if unique_teams:
        df = _dedupe_teams(df, top_n)
    else:
        df = df.head(top_n).reset_index(drop=True)

    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score"]]


def build_top20_mix(probs_df: pd.DataFrame, top_k: int = TOP_MIX, unique_teams: bool = UNIQUE_TEAMS_IN_TOP20) -> pd.DataFrame:
    if probs_df is None or probs_df.empty:
        return pd.DataFrame()

    def _all_for(col: str, label: str) -> pd.DataFrame:
        d = probs_df[CORE_COLS + ["confidence", col]].copy()
        d = d.rename(columns={col: "prob"})
        d["bet"] = label
        d["prob"] = pd.to_numeric(d["prob"], errors="coerce")
        d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
        d = d.dropna(subset=["prob"])
        d["score"] = d["prob"] * d["confidence"].fillna(0.5)
        return d[["utcDate", "league", "home", "away", "bet", "prob", "confidence", "score"]]

    parts = [
        _all_for("p_home", "HOME WIN"),
        _all_for("p_draw", "DRAW"),
        _all_for("p_away", "AWAY WIN"),
        _all_for("p_1x", "DOUBLE CHANCE 1X"),
        _all_for("p_x2", "DOUBLE CHANCE X2"),
        _all_for("p_12", "DOUBLE CHANCE 12"),
        _all_for("p_btts_yes", "BTTS YES"),
        _all_for("p_btts_no", "BTTS NO"),
        _all_for("p_over_0_5", "OVER 0.5"),
        _all_for("p_over_1_5", "OVER 1.5"),
        _all_for("p_over_2_5", "OVER 2.5"),
        _all_for("p_over_3_5", "OVER 3.5"),
        _all_for("p_home_over_0_5", "HOME TEAM OVER 0.5"),
        _all_for("p_away_over_0_5", "AWAY TEAM OVER 0.5"),
        _all_for("p_home_cs", "HOME CLEAN SHEET"),
        _all_for("p_away_cs", "AWAY CLEAN SHEET"),
        _all_for("p_home_win_to_nil", "HOME WIN TO NIL"),
        _all_for("p_away_win_to_nil", "AWAY WIN TO NIL"),
        _all_for("p_btts_yes_over_2_5", "BTTS YES & OVER 2.5"),
        _all_for("p_btts_no_under_2_5", "BTTS NO & UNDER 2.5"),
        _all_for("p_home_dnb_win", "HOME DNB (WIN PROB)"),
        _all_for("p_away_dnb_win", "AWAY DNB (WIN PROB)"),
    ]

    mix = pd.concat(parts, ignore_index=True)
    mix = mix.drop_duplicates(subset=["utcDate", "home", "away", "bet"])
    mix = mix.sort_values(["score", "prob", "utcDate"], ascending=[False, False, True]).reset_index(drop=True)

    if unique_teams:
        mix = _dedupe_teams(mix, top_k)
    else:
        mix = mix.head(top_k).reset_index(drop=True)

    mix.insert(0, "rank", range(1, len(mix) + 1))
    return mix[["rank", "utcDate", "league", "home", "away", "bet", "prob", "confidence", "score"]]


# ================= MAIN =================
def main():
    log("=== START ===")
    fd = FD(env("FOOTBALL_DATA_TOKEN"))

    today = now().date()
    f1 = today.isoformat()
    f2 = (today + timedelta(days=DAYS_AHEAD)).isoformat()
    h1 = (today - timedelta(days=HISTORY_DAYS)).isoformat()

    available_codes = set()
    try:
        for c in fd.competitions():
            code = c.get("code")
            if code:
                available_codes.add(code)
    except Exception as e:
        log(f"competitions discovery failed: {e}")

    comp_codes = list(dict.fromkeys(BASE_COMP_CODES + OPTIONAL_COMP_CODES))
    if available_codes:
        comp_codes = [c for c in comp_codes if c in available_codes] + [c for c in BASE_COMP_CODES if c not in available_codes]

    fixtures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    access_rows: List[Dict[str, Any]] = []

    for code in comp_codes:
        # Fixtures
        time.sleep(SLEEP_SECONDS)
        fx_ok, fx_n, fx_err = True, 0, ""
        try:
            fx = fd.matches(code, "SCHEDULED", f1, f2)
            fx_n = len(fx)
            log(f"{code} fixtures: {fx_n}")
            for m in fx:
                fixtures.append({
                    "league": code,
                    "utcDate": m.get("utcDate"),
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                })
        except requests.HTTPError as e:
            fx_ok = False
            status = e.response.status_code if e.response is not None else None
            fx_err = f"HTTP {status}"
            log(f"skip fixtures {code}: {fx_err}")
        except Exception as e:
            fx_ok = False
            fx_err = str(e)
            log(f"skip fixtures {code}: {fx_err}")

        # Results
        time.sleep(SLEEP_SECONDS)
        rs_ok, rs_n, rs_err = True, 0, ""
        try:
            rs = fd.matches(code, "FINISHED", h1, f1)
            rs_n = len(rs)
            log(f"{code} results: {rs_n}")
            for m in sorted(rs, key=lambda x: x.get("utcDate") or ""):
                sc = ((m.get("score") or {}).get("fullTime") or {})
                hg = sc.get("home")
                ag = sc.get("away")
                if hg is None or ag is None:
                    continue
                results.append({
                    "league": code,
                    "utcDate": m.get("utcDate") or "",
                    "home": (m.get("homeTeam") or {}).get("name"),
                    "away": (m.get("awayTeam") or {}).get("name"),
                    "hg": int(hg),
                    "ag": int(ag),
                })
        except requests.HTTPError as e:
            rs_ok = False
            status = e.response.status_code if e.response is not None else None
            rs_err = f"HTTP {status}"
            log(f"skip results {code}: {rs_err}")
        except Exception as e:
            rs_ok = False
            rs_err = str(e)
            log(f"skip results {code}: {rs_err}")

        access_rows.append({
            "code": code,
            "fixtures_ok": fx_ok,
            "fixtures_n": fx_n,
            "fixtures_err": fx_err,
            "results_ok": rs_ok,
            "results_n": rs_n,
            "results_err": rs_err,
        })

    fx_df = pd.DataFrame(fixtures)
    rs_df = pd.DataFrame(results)
    access_df = pd.DataFrame(access_rows)

    write_df("Fixtures", fx_df)
    write_df("Competitions_Access", access_df)

    # Visible tabs you want
    visible_tabs = [
        "Fixtures",
        "Team_Form",
        "Picks",
        "Top10",        # dashboard tab you will create with dropdown
        "Top20_Mix",
    ]

    if rs_df.empty:
        write_df("Team_Form", pd.DataFrame())
        write_df("Picks", pd.DataFrame())
        write_df("Top10_All", pd.DataFrame())
        write_df("Top20_Mix", pd.DataFrame())
        set_visible_tabs(visible_tabs)
        log("=== DONE (no results) ===")
        return

    # Baselines + indices
    league_base = compute_league_baselines(rs_df)
    team_idx = compute_team_indices(rs_df, league_base)

    # Write Team_Form (includes indices + sample sizes)
    team_form_df = team_idx.reset_index()
    write_df("Team_Form", team_form_df)

    if fx_df.empty:
        write_df("Picks", pd.DataFrame())
        write_df("Top10_All", pd.DataFrame())
        write_df("Top20_Mix", pd.DataFrame())
        set_visible_tabs(visible_tabs)
        log("=== DONE (no fixtures) ===")
        return

    # Build probs
    probs_rows: List[Dict[str, Any]] = []

    for _, r in fx_df.iterrows():
        league = r.get("league")
        home = r.get("home")
        away = r.get("away")

        base_home_gf = safe_float(league_base.loc[league, "home_gf"], 1.35) if league in league_base.index else 1.35
        base_away_gf = safe_float(league_base.loc[league, "away_gf"], 1.10) if league in league_base.index else 1.10

        def _get(team: str, col: str) -> float:
            try:
                return float(team_idx.loc[(league, team), col])
            except Exception:
                return 1.0

        def _get_n(team: str, col: str) -> float:
            try:
                return float(team_idx.loc[(league, team), col])
            except Exception:
                return 0.0

        ha = _get(home, "home_attack")
        hd = _get(home, "home_defense")
        aa = _get(away, "away_attack")
        ad = _get(away, "away_defense")

        n_home = _get_n(home, "n_home")
        n_away = _get_n(away, "n_away")

        # expected goals
        lh = base_home_gf * ha * ad
        la = base_away_gf * aa * hd

        # optional H2H adjustment
        dh, da = h2h_goal_adjustment(rs_df, league, home, away)
        lh = max(0.2, lh + dh)
        la = max(0.2, la + da)

        conf = confidence_score(n_home, n_away)

        p = match_probs(lh, la)
        probs_rows.append({
            "utcDate": r.get("utcDate"),
            "league": league,
            "home": home,
            "away": away,
            "lambda_home": round(lh, 3),
            "lambda_away": round(la, 3),
            "confidence": round(conf, 3),
            **p,
        })

    probs_df = pd.DataFrame(probs_rows)

    # Picks
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
            "lambda_home": row["lambda_home"],
            "lambda_away": row["lambda_away"],
            "confidence": row["confidence"],
            "pick_1x2": pick,
            "p_1x2": round(pbest, 3),
            "p_home": round(float(row["p_home"]), 3),
            "p_draw": round(float(row["p_draw"]), 3),
            "p_away": round(float(row["p_away"]), 3),
            "p_btts_yes": round(float(row["p_btts_yes"]), 3),
            "p_over_1_5": round(float(row["p_over_1_5"]), 3),
            "p_over_2_5": round(float(row["p_over_2_5"]), 3),
        })

    picks_out = pd.DataFrame(picks).sort_values(["p_1x2", "utcDate"], ascending=[False, True])
    write_df("Picks", picks_out)

    # Build Top10_All master (one table, many markets)
    markets = [
        ("p_home", "HOME WIN"),
        ("p_draw", "DRAW"),
        ("p_away", "AWAY WIN"),
        ("p_btts_yes", "BTTS YES"),
        ("p_btts_no", "BTTS NO"),
        ("p_over_0_5", "OVER 0.5"),
        ("p_over_1_5", "OVER 1.5"),
        ("p_over_2_5", "OVER 2.5"),
        ("p_over_3_5", "OVER 3.5"),
        ("p_under_1_5", "UNDER 1.5"),
        ("p_under_2_5", "UNDER 2.5"),
        ("p_under_3_5", "UNDER 3.5"),
        ("p_1x", "DOUBLE CHANCE 1X"),
        ("p_x2", "DOUBLE CHANCE X2"),
        ("p_12", "DOUBLE CHANCE 12"),
        ("p_home_dnb_win", "HOME DNB (WIN PROB)"),
        ("p_away_dnb_win", "AWAY DNB (WIN PROB)"),
        ("p_home_over_0_5", "HOME TEAM OVER 0.5"),
        ("p_away_over_0_5", "AWAY TEAM OVER 0.5"),
        ("p_home_cs", "HOME CLEAN SHEET"),
        ("p_away_cs", "AWAY CLEAN SHEET"),
        ("p_home_win_to_nil", "HOME WIN TO NIL"),
        ("p_away_win_to_nil", "AWAY WIN TO NIL"),
        ("p_btts_yes_over_2_5", "BTTS YES & OVER 2.5"),
        ("p_btts_no_under_2_5", "BTTS NO & UNDER 2.5"),
    ]

    top10_all_parts = []
    for col, label in markets:
        t = top_n_for_market(probs_df, col, label, TOP_N, unique_teams=UNIQUE_TEAMS_PER_TOP10)
        if not t.empty:
            top10_all_parts.append(t)

    top10_all = pd.concat(top10_all_parts, ignore_index=True) if top10_all_parts else pd.DataFrame()
    write_df("Top10_All", top10_all)

    # Top20 mix unique per team
    mix_df = build_top20_mix(probs_df, TOP_MIX, unique_teams=UNIQUE_TEAMS_IN_TOP20)
    write_df("Top20_Mix", mix_df)

    # Make only your key tabs visible (Top10_All will be used by formulas; keep it hidden)
    set_visible_tabs(visible_tabs)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
