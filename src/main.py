from __future__ import annotations
import os
import pandas as pd
from .utils import env, utc_now_iso, load_json
from .odds_api import OddsApi, flatten_odds
from .football_data import read_csv_url, normalize_results
from .model import build_team_form, build_predictions
from .value import build_value_bets
from .sheets import _client_from_service_account_json, write_df, append_log

def pick_target_sports(sports: list[dict], wanted: list[str]) -> list[dict]:
    wanted_l = [w.lower() for w in wanted]
    out=[]
    for s in sports:
        title = (s.get("title") or "").lower()
        if any(w in title for w in wanted_l):
            out.append(s)
    return out

def main():
    cfg = load_json("config/leagues.json")
    region = env("REGION", cfg.get("region","uk"))
    bankroll = float(os.getenv("BANKROLL") or 100.0)
    kelly_fraction = float(os.getenv("KELLY_FRACTION") or 0.25)
    min_edge = float(os.getenv("MIN_EDGE") or 0.03)

    odds_key = env("ODDS_API_KEY")
    sheet_id = env("SHEET_ID")
    sa_json = env("GOOGLE_SERVICE_ACCOUNT_JSON")

    client = _client_from_service_account_json(sa_json)
    sh = client.open_by_key(sheet_id)

    append_log(sh, f"Run started {utc_now_iso()} (region={region})")

    # 1) Historical results (form)
    results_all=[]
    for code, url in cfg["football_data_csv"].items():
        try:
            df = read_csv_url(url)
            results_all.append(normalize_results(df, code))
        except Exception as e:
            append_log(sh, f"Failed to fetch {code}: {e}")
    results = pd.concat(results_all, ignore_index=True) if results_all else pd.DataFrame()
    write_df(sh, "Results_History", results.tail(2000))

    form = build_team_form(results, window=int(cfg.get("rolling_window_matches",12)))
    write_df(sh, "Team_Form", form.sort_values("team"))

    # 2) Odds events
    api = OddsApi(api_key=odds_key, region=region)
    sports = api.list_sports()
    targets = pick_target_sports(sports, cfg["target_leagues"])

    append_log(sh, f"Matched {len(targets)} competitions from Odds API list")

    events_all=[]
    for s in targets:
        if not s.get("active", True):
            continue
        skey = s.get("key")
        try:
            events = api.get_odds(skey, cfg["odds_markets"])
            events_all.extend(events)
        except Exception as e:
            append_log(sh, f"Failed odds for {skey}: {e}")

    odds_df = flatten_odds(events_all)
    write_df(sh, "Odds_Snapshot", odds_df)

    # Fixtures view from odds events
    if odds_df.empty:
        fixtures = pd.DataFrame(columns=["competition","commence_time","home_team","away_team"])
    else:
        fixtures = odds_df[["competition","commence_time","home_team","away_team"]].drop_duplicates()
        fixtures = fixtures.sort_values(["competition","commence_time"])
    write_df(sh, "Fixtures", fixtures)

    # 3) Model predictions
    model = build_predictions(fixtures, form) if not fixtures.empty else pd.DataFrame()
    write_df(sh, "Model_Probs", model)

    # 4) Value bets (no HT/FT)
    value = build_value_bets(
        odds_best=odds_df,
        model=model,
        bankroll=bankroll,
        kelly_mult=kelly_fraction,
        min_edge=min_edge
    )
    write_df(sh, "Value_Bets", value)

    append_log(sh, f"Run finished {utc_now_iso()} | fixtures={len(fixtures)} | value_bets={len(value)}")

if __name__ == "__main__":
    main()
