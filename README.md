# Football Betting Bot (Upgraded – Free Sources)

Automated value-betting system using only free data sources.

## Features
- football-data.org + football-data.co.uk CSVs
- Free xG from Understat + FBref (soccerdata + Scrapling fallback)
- The Odds API with strong Paddy Power / BoyleSports preference
- Optional Scrapling stealth enrichment for named books
- Poisson model blended with xG
- More leagues and markets
- Google Sheets output + history / accuracy tracking

## Setup

### Secrets (GitHub Actions)
- `ODDS_API_KEY`
- `FOOTBALL_DATA_TOKEN`
- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `SHEET_ID`

### Local install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
scrapling install
