from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials


# ----------------------------
# Requested competitions (Odds API sport keys)
# ----------------------------
TARGET_SPORT_KEYS = [
    # Top 5
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    # UEFA
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league",
    # UK lower leagues
    "soccer_efl_champ",
    "soccer_england_league1",
    "soccer_england_league2",
    # Scotland
    "soccer_spl",
    # Spain 2
    "soccer_spain_segunda_division",
]

# We will try these market sets in order (fallback on 422)
MARKET_FALLBACKS = [
    "h2h,totals,btts",
    "h2h,totals",
    "h2h",
]

DEFAULT_TOTAL_LINE = 2.5


# ----------------------------
# Utilities
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_json(s: str) -> Any:
    return json.loads(s)


# ----------------------------
# Google Sheets helpers
# ----------------------------
def _gs_client() -> gspread.Client:
    raw = env("GOOGLE_SERVICE_ACCOUNT_JSON")
    info = load_json(raw)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def _open_sheet(sheet_id: str):
    return _gs_client().open_by_key(sheet_id)


def write_df(sheet_name: str, df: pd.DataFrame) -> None:
    sheet = _open_sheet(env("SHEET_ID"))
    try:
        ws = sheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=sheet_name, rows=1000, cols=26)

    if df is None:
        df = pd.DataFrame()

    if df.empty:
        values: List[List[Any]] = [["(no data)"]]
    else:
        values = [list(df.columns)] + df.fillna("").values.tolist()

    ws.clear()
    ws.update(values)


def append_log(row: Dict[str, Any]) -> None:
    sheet = _open_sheet(env("SHEET_ID"))
    try:
        ws = sheet.worksheet("Run_Log")
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title="Run_Log", rows=2000, cols=30)
        ws.append_row(list(row.keys()))

    header = ws.row_values(1)
    if not header:
        ws.append_row(list(row.keys()))
        header = list(row.keys())

    # Add missing columns if new keys appear
    missing = [k for k in row.keys() if k not in header]
    if missing:
        header.extend(missing)
        ws.clear()
        ws.append_row(header)

    ws.append_row([row.get(k, "") for k in header])


# ----------------------------
# Odds API client
# ----------------------------
@dataclass
class OddsApi:
    api_key: str
    region: str = "uk"
    odds_format: str = "decimal"
    base: str = "https://api.the-odds-api.com/v4"

    def list_sports(self) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base}/sports", params={"apiKey": self.api_key}, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_odds_for_sport(self, sport_key: str, markets: str) -> List[Dict[str, Any]]:
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": markets,
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
        }
        r = requests.get(f"{self.base}/sports/{sport_key}/odds", params=params, timeout=30)
        r.raise_for_status()
        return r.json()


# ----------------------------
# Selection + flattening
# ----------------------------
def pick_target_sports(sports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = {s.get("key") for s in sports if isinstance(s, dict)}
    missing = [k for k in TARGET_SPORT_KEYS if k not in keys]
    if missing:
        _log(f"⚠️ Missing from Odds API sports list (will skip): {missing}")
    return [s for s in sports if s.get("key") in TARGET_SPORT_KEYS]


def flatten_odds(events: List[Dict[str, Any]], sport_key: str, sport_title: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ev in events or []:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")

        for bk in ev.get("bookmakers", []) or []:
            bookmaker = bk.get("title")
            last_update = bk.get("last_update")

            for m in bk.get("markets", []) or []:
                market = m.get("key")
                for o in m.get("outcomes", []) or []:
                    rows.append(
                        {
                            "sport_key": sport_key,
                            "sport_title": sport_title,
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": bookmaker,
                            "last_update": last_update,
                            "market": market,
                            "selection": o.get("name"),
                            "price": o.get("price"),
                            "point": o.get("point", ""),
                        }
                    )
    return pd.DataFrame(rows)


# ----------------------------
# Very simple model: Poisson goals from historical CSV (optional)
# If history missing, we still write Odds + Fixtures, and Value_Bets may be empty.
# ----------------------------
def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _match_probs_from_lambdas(lh: float, la: float, max_goals: int = 10) -> Dict[str, float]:
    ph = [_poisson_pmf(lh, i) for i in range(max_goals + 1)]
    pa = [_poisson_pmf(la, j) for j in range(max_goals + 1)]

    p_home = p_draw = p_away = 0.0
    p_btts_yes = 0.0
    p_over_25 = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i > 0 and j > 0:
                p_btts_yes += p
            if (i + j) > 2:
                p_over_25 += p

    return {
        "lambda_home": lh,
        "lambda_away": la,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": 1 - p_btts_yes,
        "p_over_2_5": p_over_25,
        "p_under_2_5": 1 - p_over_25,
    }


def normalize_results() -> pd.DataFrame:
    # keep your existing URLs if you like; this is optional
    urls = [
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E3.csv",
        "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
    ]
    frames = []
    for u in urls:
        try:
            df = pd.read_csv(u)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(df.columns):
        return pd.DataFrame()

    df = df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
    df = df.dropna(subset=["hg", "ag"])
    df["hg"] = pd.to_numeric(df["hg"], errors="coerce")
    df["ag"] = pd.to_numeric(df["ag"], errors="coerce")
    df = df.dropna(subset=["hg", "ag"])
    return df[["home", "away", "hg", "ag"]]


def build_team_form(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    home = results.groupby("home")[["hg", "ag"]].mean().rename(columns={"hg": "gf_home", "ag": "ga_home"})
    away = results.groupby("away")[["ag", "hg"]].mean().rename(columns={"ag": "gf_away", "hg": "ga_away"})
    form = home.join(away, how="outer").fillna(0.0)
    form["gf"] = (form["gf_home"] + form["gf_away"]) / 2.0
    form["ga"] = (form["ga_home"] + form["ga_away"]) / 2.0
    form["attack"] = (form["gf"] / form["gf"].mean()) if form["gf"].mean() > 0 else 1.0
    form["defense"] = (form["ga"] / form["ga"].mean()) if form["ga"].mean() > 0 else 1.0
    form = form.reset_index().rename(columns={"index": "team"})
    return form


def build_predictions(fixtures: pd.DataFrame, team_form: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()

    tf = team_form.set_index("team") if not team_form.empty else None
    base_home, base_away = 1.45, 1.20

    rows = []
    for _, r in fixtures.iterrows():
        home = r["home_team"]
        away = r["away_team"]

        if tf is not None and home in tf.index and away in tf.index:
            lh = base_home * float(tf.loc[home, "attack"]) * float(tf.loc[away, "defense"])
            la = base_away * float(tf.loc[away, "attack"]) * float(tf.loc[home, "defense"])
        else:
            lh, la = base_home, base_away

        probs = _match_probs_from_lambdas(lh, la)
        rows.append(
            {
                "sport_key": r["sport_key"],
                "sport_title": r["sport_title"],
                "event_id": r["event_id"],
                "commence_time": r["commence_time"],
                "home_team": home,
                "away_team": away,
                **probs,
            }
        )

    return pd.DataFrame(rows)


def build_value_bets(odds_df: pd.DataFrame, probs_df: pd.DataFrame, bankroll: float, kelly_fraction: float, min_edge: float) -> pd.DataFrame:
    if odds_df.empty or probs_df.empty:
        return pd.DataFrame()

    pmap = probs_df.set_index("event_id")
    rows = []

    for _, o in odds_df.iterrows():
        event_id = o["event_id"]
        if event_id not in pmap.index:
            continue

        market = o["market"]
        price = o["price"]
        if price in ("", None) or pd.isna(price):
            continue
        price = float(price)

        model = pmap.loc[event_id]

        if market == "h2h":
            sel = str(o["selection"]).strip().lower()
            if sel == str(o["home_team"]).strip().lower():
                p = float(model["p_home"])
            elif sel == str(o["away_team"]).strip().lower():
                p = float(model["p_away"])
            else:
                p = float(model["p_draw"])

        elif market == "btts":
            sel = str(o["selection"]).strip().lower()
            p = float(model["p_btts_yes"] if sel in ("yes", "y") else model["p_btts_no"])

        elif market == "totals":
            # Only 2.5 for simplicity
            try:
                line = float(o.get("point", DEFAULT_TOTAL_LINE) or DEFAULT_TOTAL_LINE)
            except Exception:
                line = DEFAULT_TOTAL_LINE
            if abs(line - 2.5) > 1e-6:
                continue
            sel = str(o["selection"]).strip().lower()
            p = float(model["p_over_2_5"] if "over" in sel else model["p_under_2_5"])

        else:
            continue

        implied = 1.0 / price
        edge = p - implied
        ev_roi = (p * price) - 1.0
        if ev_roi <= 0 or edge < min_edge:
            continue

        b = price - 1.0
        kelly = (p * b - (1 - p)) / b if b > 0 else 0.0
        stake = max(0.0, bankroll * kelly_fraction * kelly)

        rows.append(
            {
                "commence_time": o["commence_time"],
                "sport_title": o["sport_title"],
                "event_id": event_id,
                "home_team": o["home_team"],
                "away_team": o["away_team"],
                "bookmaker": o["bookmaker"],
                "market": market,
                "selection": o["selection"],
                "line": o.get("point", ""),
                "odds": price,
                "model_prob": round(p, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "ev": round(ev_roi, 4),
                "stake": round(stake, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["ev", "edge"], ascending=[False, False]).reset_index(drop=True)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ts = utc_now_iso()
    _log(f"=== RUN {ts} ===")

    api = OddsApi(api_key=env("ODDS_API_KEY"), region=os.getenv("REGION", "uk"))
    bankroll = float(os.getenv("BANKROLL", "100"))
    kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.25"))
    min_edge = float(os.getenv("MIN_EDGE", "0.03"))

    sports = api.list_sports()
    targets = pick_target_sports(sports)
    _log(f"Sports total={len(sports)} target_matched={len(targets)}")

    odds_frames: List[pd.DataFrame] = []
    fixtures_frames: List[pd.DataFrame] = []
    total_events = 0
    stopped_on_429 = False

    for s in targets:
        key = s["key"]
        title = s.get("title", key)

        events: List[Dict[str, Any]] = []
        used_markets = ""

        for markets in MARKET_FALLBACKS:
            try:
                used_markets = markets
                events = api.get_odds_for_sport(key, markets=markets)
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None

                # IMPORTANT: don’t print URLs (they include apiKey)
                if status == 429:
                    _log(f"⛔ 429 rate limit/quota hit while fetching {key}. Stopping further requests.")
                    stopped_on_429 = True
                    events = []
                    break

                if status == 422:
                    _log(f"⚠️ 422 for {key} with markets='{markets}'. Trying fallback markets…")
                    continue

                _log(f"❌ HTTP error for {key} (status={status}). Skipping this league.")
                events = []
                break
            except Exception as e:
                _log(f"❌ Unexpected error for {key}: {e}")
                events = []
                break

        if stopped_on_429:
            break

        total_events += len(events or [])
        df = flatten_odds(events, sport_key=key, sport_title=title)
        odds_frames.append(df)

        if not df.empty:
            fx = df[["sport_key", "sport_title", "event_id", "commence_time", "home_team", "away_team"]].drop_duplicates()
            fixtures_frames.append(fx)

        _log(f"{key}: markets_used='{used_markets}' events={len(events or [])} odds_rows={len(df)}")

    odds_df = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()
    fixtures_df = pd.concat(fixtures_frames, ignore_index=True) if fixtures_frames else pd.DataFrame()

    # Always write tabs (so you never see "no data" without context)
    write_df("Odds_Snapshot", odds_df if not odds_df.empty else pd.DataFrame(columns=[
        "sport_key","sport_title","event_id","commence_time","home_team","away_team",
        "bookmaker","last_update","market","selection","price","point"
    ]))
    write_df("Fixtures", fixtures_df if not fixtures_df.empty else pd.DataFrame(columns=[
        "sport_key","sport_title","event_id","commence_time","home_team","away_team"
    ]))

    results_df = normalize_results()
    team_form = build_team_form(results_df) if not results_df.empty else pd.DataFrame()
    probs_df = build_predictions(fixtures_df, team_form) if not fixtures_df.empty else pd.DataFrame()
    write_df("Model_Probs", probs_df if not probs_df.empty else pd.DataFrame(columns=[
        "sport_key","sport_title","event_id","commence_time","home_team","away_team",
        "lambda_home","lambda_away","p_home","p_draw","p_away",
        "p_btts_yes","p_btts_no","p_over_2_5","p_under_2_5"
    ]))

    value_df = build_value_bets(odds_df, probs_df, bankroll, kelly_fraction, min_edge)
    write_df("Value_Bets", value_df if not value_df.empty else pd.DataFrame(columns=[
        "commence_time","sport_title","event_id","home_team","away_team","bookmaker",
        "market","selection","line","odds","model_prob","implied_prob","edge","ev","stake"
    ]))

    append_log({
        "ts": ts,
        "sports_total": len(sports),
        "sports_target": len(targets),
        "events_total": total_events,
        "odds_rows": int(len(odds_df)),
        "fixtures_rows": int(len(fixtures_df)),
        "results_rows": int(len(results_df)),
        "probs_rows": int(len(probs_df)),
        "value_rows": int(len(value_df)),
        "stopped_on_429": stopped_on_429,
        "note": "429 hit (quota/rate limit) — reduce leagues or wait for reset" if stopped_on_429 else "OK",
    })

    _log("=== DONE ===")


if __name__ == "__main__":
    main()
