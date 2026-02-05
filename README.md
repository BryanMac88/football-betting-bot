# Football Betting Bot (Free) — GitHub Actions + Google Sheets

Fully automated daily run (10:00 UTC) that:
- Pulls **odds** from **The Odds API** (free tier) for supported markets: **1X2 (h2h)**, **Totals (O/U goals)**, **BTTS**, and **Cards O/U** (when available).
- Pulls **historical match data** (incl. cards & fouls columns when available) from **football-data.co.uk** (free CSV).
- Builds simple probabilities (Poisson goals + basic totals for cards/fouls).
- Writes everything into **Google Sheets** tabs, including a **Value_Bets** shortlist (EV + edge + fractional Kelly stake).

> Note on bookmakers: Paddy Power / BoyleSports are UK/IE books. The Odds API uses **regions=uk** and returns odds from many UK-facing bookmakers depending on availability.

## 1) Create a Google Sheet
Create a new Google Sheet and copy its ID (the long string in the URL).

Share the sheet with your Google **service account email** as Editor.

## 2) Create a Google Cloud Service Account (free)
- Enable **Google Sheets API**
- Create a service account and generate a **JSON key**
- Store the full JSON as a GitHub secret: `GOOGLE_SERVICE_ACCOUNT_JSON`

## 3) Get free API keys
- The Odds API key: https://the-odds-api.com/
- (Optional) football-data.org key is **not required** here; we use football-data.co.uk for historical stats.

Add secrets:
- `ODDS_API_KEY`
- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `SHEET_ID`

Optional config secrets:
- `BANKROLL` (default: 100)
- `KELLY_FRACTION` (default: 0.25)
- `MIN_EDGE` (default: 0.03)  # 3% edge threshold
- `REGION` (default: uk)

## 4) Choose leagues
Edit `config/leagues.json` to control which competitions are included.

This repo includes:
- Top 5 European leagues (ENG, ESP, GER, ITA, FRA)
- UCL, UEL, UECL (odds only; historical form uses domestic leagues)
- English Championship, League 1, League 2
- Spanish Segunda (and a placeholder for "Spain 3" if your odds provider lists it)
- SPL (Scottish Premiership)

## 5) GitHub Actions schedule
Runs daily at **10:00 UTC**:
`.github/workflows/daily.yml`

If you need **10:00 Dublin year-round**, set the cron to 10:00 in winter and 09:00 in summer, or run at 10:00 UTC and accept the 1h shift during DST.

## What gets written to Google Sheets
Tabs:
- `Run_Log`
- `Fixtures` (upcoming odds events)
- `Odds_Snapshot` (raw best odds per market/selection/line)
- `Team_Form` (rolling averages)
- `Model_Probs` (model probabilities & fair odds)
- `Value_Bets` (filtered best bets with EV + stake)

## Local run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ODDS_API_KEY="..."
export GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
export SHEET_ID="..."
python -m src.main
```

## Disclaimer
This is a simple statistical model and cannot guarantee profit. Betting carries risk.
