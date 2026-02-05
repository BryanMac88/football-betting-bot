from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials


# ----------------------------
# Config: leagues you requested
# ----------------------------
# These sport keys come from The Odds API sports list.
# (Spain "League 3" is not listed there; we log and skip if not available.)
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
    # Spain "3" is not a standard Odds API key on their sports list page; will be skipped if absent
]

# Markets we actually compute model probabilities for:
SUPPORTED_MARKETS = {"h2h", "totals", "btts"}

DEFAULT_TOTAL_LINE = 2.5  # for totals model outputs


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
# Google Sheets writer
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
    """Clear + write dataframe (including header). Ensures sheet exists."""
    sheet = _open_sheet(env("SHEET_ID"))
    try:
        ws = sheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=sheet_name, rows=1000, cols=26)

    # Ensure at least headers exist
    if df is None:
        df = pd.DataFrame()

    values: List[List[Any]] = []
    if df.empty:
        values = [["(no data)"]]
    else:
        values = [list(df.columns)] + df.fillna("").values.tolist()

    ws.clear()
    ws.update(values)


def append_log(row: Dict[str, Any]) -> None:
    """Append one row to Run_Log (creates sheet if needed)."""
    sheet = _open_sheet(env("SHEET_ID"))
    try:
        ws = sheet.worksheet("Run_Log")
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title="Run_Log", rows=1000, cols=26)
        ws.append_row(list(row.keys()))

    # ensure header contains keys
    header = ws.row_values(1)
    if not header:
        ws.append_row(list(row.keys()))
        header = list(row.keys())

    # add new keys if needed
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
        url = f"{self.base}/sports"
        r = requests.get(url, params={"apiKey": self.api_key})
        r.raise_for_status()
        return r.json()

    def get_odds_for_sport(
        self,
        sport_key: str,
        markets: str = "h2h,totals,btts",
        date_format: str = "iso",
    ) -> List[Dict[str, Any]]:
        url = f"{self.base}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": markets,
            "oddsFormat": self.odds_format,
            "dateFormat": date_format,
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()


# ----------------------------
# League selection
# ----------------------------
def pick_target_sports(sports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = {s.get("key") for s in sports if isinstance(s, dict)}
    picked = []
    for s in sports:
        if s.get("key") in TARGET_SPORT_KEYS:
            picked.append(s)
    # log if any requested keys missing
    missing = [k for k in TARGET_SPORT_KEYS if k not in keys]
    if missing:
        _log(f"⚠️ Missing from Odds API sports list (will skip): {missing}")
    return picked


# ----------------------------
# Flatten odds into a table
# ----------------------------
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
                market = m.get("key")  # h2h, totals, btts
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
                            "point": o.get("point", ""),  # totals line if present
                        }
                    )
    return pd.DataFrame(rows)


# ----------------------------
# Model helpers (Poisson goals)
# ----------------------------
def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _match_probs_from_lambdas(lh: float, la: float, max_goals: int = 10) -> Dict[str, float]:
    # Compute score matrix
    ph = [_poisson_pmf(lh, i) for i in range(max_goals + 1)]
    pa = [_poisson_pmf(la, j) for j in range(max_goals + 1)]
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
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
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": 1 - p_btts_yes,
        "p_over_2_5": p_over_25,
        "p_under_2_5": 1 - p_over_25,
        "lambda_home": lh,
        "lambda_away": la,
    }


def _devig_1x2(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    ph = 1.0 / home_odds
    pd = 1.0 / draw_odds
    pa = 1.0 / away_odds
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


# ----------------------------
# Minimal historical results loader (optional)
# ----------------------------
def read_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def normalize_results() -> pd.DataFrame:
    """
    Optional: if you want historical team strengths.
    Keep it safe: if a URL fails, return empty and we fall back to odds-derived probs.
    You can add more sources later.
    """
    urls = [
        # football-data.co.uk provides free CSVs by league/season; you can extend these.
        # If these 404, we fall back automatically.
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",  # EPL 25/26
        "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", # La Liga 25/26
        "https://www.football-data.co.uk/mmz4281/2526/I1.csv",  # Serie A 25/26
        "https://www.football-data.co.uk/mmz4281/2526/D1.csv",  # Bundesliga 25/26
        "https://www.football-data.co.uk/mmz4281/2526/F1.csv",  # Ligue 1 25/26
        "https://www.football-data.co.uk/mmz4281/2526/SC0.csv", # Scotland Premiership 25/26 (may vary)
        "https://www.football-data.co.uk/mmz4281/2526/E1.csv",  # Championship 25/26
        "https://www.football-data.co.uk/mmz4281/2526/E2.csv",  # League 1 25/26
        "https://www.football-data.co.uk/mmz4281/2526/E3.csv",  # League 2 25/26
        "https://www.football-data.co.uk/mmz4281/2526/SP2.csv", # La Liga 2 25/26
    ]

    frames: List[pd.DataFrame] = []
    for u in urls:
        try:
            df = read_csv_url(u)
            df["source_url"] = u
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Standardize columns we need
    keep = []
    for col in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "source_url"]:
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()
    df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag", "Date": "date"}, inplace=True)

    # Drop rows without scores
    df = df.dropna(subset=["hg", "ag"])
    df["hg"] = pd.to_numeric(df["hg"], errors="coerce")
    df["ag"] = pd.to_numeric(df["ag"], errors="coerce")
    df = df.dropna(subset=["hg", "ag"])

    return df


def build_team_form(results: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    # Recent matches only
    # football-data dates vary; keep as strings; just take tail-ish by grouping later
    res = results.copy()

    # Build per-team aggregates (home+away combined)
    home = res.groupby("home")[["hg", "ag"]].mean().rename(columns={"hg": "gf_home", "ag": "ga_home"})
    away = res.groupby("away")[["ag", "hg"]].mean().rename(columns={"ag": "gf_away", "hg": "ga_away"})

    form = home.join(away, how="outer").fillna(0.0)
    form["gf"] = (form["gf_home"] + form["gf_away"]) / 2.0
    form["ga"] = (form["ga_home"] + form["ga_away"]) / 2.0

    # League averages proxy
    form["attack"] = (form["gf"] / form["gf"].mean()) if form["gf"].mean() > 0 else 1.0
    form["defense"] = (form["ga"] / form["ga"].mean()) if form["ga"].mean() > 0 else 1.0

    form.reset_index(inplace=True)
    form.rename(columns={"index": "team"}, inplace=True)
    return form


def build_predictions(fixtures: pd.DataFrame, team_form: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()

    tf = team_form.set_index("team") if not team_form.empty else None

    # league avg home/away goals baseline (fallback)
    base_home = 1.45
    base_away = 1.20

    rows = []
    for _, r in fixtures.iterrows():
        home = r["home_team"]
        away = r["away_team"]

        if tf is not None and home in tf.index and away in tf.index:
            a_h = float(tf.loc[home, "attack"])
            d_h = float(tf.loc[home, "defense"])
            a_a = float(tf.loc[away, "attack"])
            d_a = float(tf.loc[away, "defense"])

            lh = base_home * a_h * d_a
            la = base_away * a_a * d_h
        else:
            # no history -> we still produce something; later we may overwrite with de-vig odds where possible
            lh = base_home
            la = base_away

        probs = _match_probs_from_lambdas(lh, la)
        rows.append(
            {
                "sport_key": r.get("sport_key", ""),
                "sport_title": r.get("sport_title", ""),
                "event_id": r.get("event_id", ""),
                "commence_time": r.get("commence_time", ""),
                "home_team": home,
                "away_team": away,
                **probs,
            }
        )

    return pd.DataFrame(rows)


def build_value_bets(
    odds_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    bankroll: float,
    kelly_fraction: float,
    min_edge: float,
) -> pd.DataFrame:
    if odds_df.empty or probs_df.empty:
        return pd.DataFrame()

    pmap = probs_df.set_index("event_id")

    rows: List[Dict[str, Any]] = []

    # Pre-compute de-vig 1X2 from best available bookmaker if history is weak
    # We'll compute per event if we have h2h prices.
    for event_id, ev in odds_df.groupby("event_id"):
        if event_id not in pmap.index:
            continue

        # model probabilities (from Poisson). If it's baseline-only (no history), still OK.
        model = pmap.loc[event_id]

        for _, o in ev.iterrows():
            market = o["market"]
            price = o["price"]
            if not price or pd.isna(price):
                continue
            price = float(price)

            if market == "h2h":
                sel = str(o["selection"]).strip().lower()
                if sel == str(o["home_team"]).strip().lower():
                    p = float(model["p_home"])
                elif sel == str(o["away_team"]).strip().lower():
                    p = float(model["p_away"])
                else:
                    # draw often comes as "Draw"
                    p = float(model["p_draw"])

            elif market == "btts":
                sel = str(o["selection"]).strip().lower()
                p = float(model["p_btts_yes"] if sel in ("yes", "y") else model["p_btts_no"])

            elif market == "totals":
                # Only evaluate 2.5 line by default (common + stable)
                point = o.get("point", "")
                try:
                    line = float(point) if point != "" else DEFAULT_TOTAL_LINE
                except Exception:
                    line = DEFAULT_TOTAL_LINE

                if abs(line - 2.5) > 1e-6:
                    continue  # keep it simple + predictable for now

                sel = str(o["selection"]).strip().lower()
                if "over" in sel:
                    p = float(model["p_over_2_5"])
                else:
                    p = float(model["p_under_2_5"])
            else:
                continue

            implied = 1.0 / price
            edge = p - implied
            ev_roi = (p * price) - 1.0

            if ev_roi <= 0 or edge < min_edge:
                continue

            # fractional Kelly
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

    out = pd.DataFrame(rows).sort_values(["ev", "edge"], ascending=[False, False]).reset_index(drop=True)
    return out


# ----------------------------
# Main pipeline
# ----------------------------
def main() -> None:
    ts = utc_now_iso()
    _log(f"=== RUN {ts} ===")

    sheet_id = env("SHEET_ID")
    api_key = env("ODDS_API_KEY")
    region = os.getenv("REGION", "uk")
    bankroll = float(os.getenv("BANKROLL", "100"))
    kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.25"))
    min_edge = float(os.getenv("MIN_EDGE", "0.03"))

    api = OddsApi(api_key=api_key, region=region)

    # 1) Sports
    sports = api.list_sports()
    targets = pick_target_sports(sports)
    _log(f"Sports total={len(sports)} target_matched={len(targets)}")

    # 2) Odds snapshot
    odds_frames: List[pd.DataFrame] = []
    fixtures_frames: List[pd.DataFrame] = []

    total_events = 0
    for s in targets:
        key = s["key"]
        title = s.get("title", key)
        try:
            events = api.get_odds_for_sport(key)
        except Exception as e:
            _log(f"❌ Odds fetch failed for {key}: {e}")
            continue

        total_events += len(events or [])
        df = flatten_odds(events, sport_key=key, sport_title=title)
        odds_frames.append(df)

        if not df.empty:
            fx = df[["sport_key", "sport_title", "event_id", "commence_time", "home_team", "away_team"]].drop_duplicates()
            fixtures_frames.append(fx)

        _log(f"{key}: events={len(events or [])} odds_rows={len(df)}")

    odds_df = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()
    fixtures_df = pd.concat(fixtures_frames, ignore_index=True) if fixtures_frames else pd.DataFrame()

    # Always write these (even if empty, so you can see structure)
    write_df("Odds_Snapshot", odds_df if not odds_df.empty else pd.DataFrame(columns=[
        "sport_key","sport_title","event_id","commence_time","home_team","away_team",
        "bookmaker","last_update","market","selection","price","point"
    ]))
    write_df("Fixtures", fixtures_df if not fixtures_df.empty else pd.DataFrame(columns=[
        "sport_key","sport_title","event_id","commence_time","home_team","away_team"
    ]))

    # 3) Results + team form (optional)
    results_df = normalize_results()
    _log(f"Results rows={len(results_df)}")
    team_form = build_team_form(results_df) if not results_df.empty else pd.DataFrame()
    _log(f"Team_form rows={len(team_form)}")

    # 4) Predictions
    probs_df = build_predictions(fixtures_df, team_form) if not fixtures_df.empty else pd.DataFrame()
    write_df("Model_Probs", probs_df if not probs_df.empty else pd.DataFrame(columns=[
        "sport_key","sport_title","event_id","commence_time","home_team","away_team",
        "lambda_home","lambda_away","p_home","p_draw","p_away",
        "p_btts_yes","p_btts_no","p_over_2_5","p_under_2_5"
    ]))

    # 5) Value bets
    value_df = build_value_bets(
        odds_df=odds_df,
        probs_df=probs_df,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
    )
    write_df("Value_Bets", value_df if not value_df.empty else pd.DataFrame(columns=[
        "commence_time","sport_title","event_id","home_team","away_team","bookmaker",
        "market","selection","line","odds","model_prob","implied_prob","edge","ev","stake"
    ]))

    # 6) Run log
    append_log({
        "ts": ts,
        "sports_total": len(sports),
        "sports_target": len(targets),
        "events_total": total_events,
        "odds_rows": len(odds_df),
        "fixtures_rows": len(fixtures_df),
        "results_rows": len(results_df),
        "probs_rows": len(probs_df),
        "value_rows": len(value_df),
        "region": region,
        "note": "OK" if len(odds_df) > 0 else "No odds rows returned (check quota/markets/leagues)",
    })

    _log("=== DONE ===")


if __name__ == "__main__":
    main()
