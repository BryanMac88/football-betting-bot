# Football Betting Bot

Automated value betting bot for football matches using The Odds API and Google Sheets.

## Setup

1. Add these **Secrets** in GitHub Settings → Secrets and variables → Actions:
   - `ODDS_API_KEY`
   - `GOOGLE_SERVICE_ACCOUNT_JSON`
   - `SHEET_ID`

2. Run manually from the **Actions** tab.

## Project Structure
- `src/` → Main Python code
- `config/` → League settings
- `.github/workflows/` → Daily automation
