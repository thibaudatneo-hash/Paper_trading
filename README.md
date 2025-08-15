# XTB-like Paper Trading (Streamlit)

A simple paper-trading dashboard for daily returns & live-ish updates (via polling).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Choose start date to view returns.
- Track daily and cumulative returns for a preloaded universe (PLTR, SMCI, VST, NVDA, AVGO, ANET, CEG, NRG, META).
- Paper trading: enter a notional amount, select allocation (equal-weight or custom weights), and "Place trades".
- Add multiple orders over time; portfolio and orders persist during the session (browser tab).
- Live-ish updates: app auto-refreshes every 60s; you can change the interval in the sidebar.
- All timestamps shown in your local timezone (default Europe/Paris).

⚠️ Data from Yahoo Finance via `yfinance` can be delayed; not true exchange-real-time.
