
import json
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

PARIS = pytz.timezone("Europe/Paris")
st.set_page_config(page_title="Paper Trading - Live Returns", page_icon="📈", layout="wide")

def to_paris(dt):
    if dt.tzinfo is None:
        return PARIS.localize(dt)
    return dt.astimezone(PARIS)

def now_paris():
    return to_paris(datetime.utcnow().replace(tzinfo=pytz.utc))

def load_universe():
    with open("tickers.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def fetch_history(tickers, start_date):
    df = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.loc[:, [t for t in tickers if t in df.columns]]
    return df

def latest_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            p = info.get("last_price") or info.get("lastPrice") or np.nan
        except Exception:
            p = np.nan
        prices[t] = p
    return pd.Series(prices, name="last")

def place_order(amount, weights, price_series, when=None):
    if when is None:
        when = now_paris()
    alloc = amount * weights
    shares = (alloc / price_series).fillna(0).replace([np.inf, -np.inf], 0)
    shares = np.floor(shares)
    cost = (shares * price_series).sum()
    return {
        "timestamp": when.isoformat(),
        "amount": float(amount),
        "weights": weights.to_dict(),
        "filled_prices": price_series.to_dict(),
        "shares": shares.astype(int).to_dict(),
        "cost": float(cost),
    }

def portfolio_positions(orders):
    pos = {}
    for o in orders:
        for t, q in o["shares"].items():
            pos[t] = pos.get(t, 0) + int(q)
    return pd.Series(pos, name="qty", dtype=int)

def portfolio_cost_basis(orders):
    cp = {}
    for o in orders:
        for t, q in o["shares"].items():
            price = o["filled_prices"].get(t, np.nan)
            if q > 0 and not np.isnan(price):
                if t not in cp:
                    cp[t] = {"qty": 0, "cost": 0.0}
                cp[t]["qty"] += int(q)
                cp[t]["cost"] += float(q) * float(price)
    avg = {}
    for t, v in cp.items():
        avg[t] = (v["cost"] / v["qty"]) if v["qty"] > 0 else np.nan
    return pd.Series(avg, name="avg_cost")

def compute_equity_curve(history, positions):
    common = [t for t in positions.index if t in history.columns and positions[t] > 0]
    if not common:
        return pd.Series(dtype=float)
    qty = positions[common].astype(float).values
    values = history[common].multiply(qty, axis=1).sum(axis=1)
    return values

def compute_returns(series):
    rets = series.pct_change().fillna(0.0)
    cum = (1 + rets).cumprod() - 1.0
    return rets, cum

if "orders" not in st.session_state:
    st.session_state.orders = []

if "autorefresh" not in st.session_state:
    st.session_state.autorefresh = 60000

universe = load_universe()
available_tickers = universe["ticker"].tolist()

st.sidebar.header("⚙️ Paramètres")
start_date = st.sidebar.date_input("Date de départ (pour les rendements)", value=(now_paris() - timedelta(days=120)).date(), max_value=now_paris().date())
selected = st.sidebar.multiselect("Univers d'actions", options=available_tickers, default=available_tickers)
refresh_ms = st.sidebar.number_input("Auto-refresh (ms)", min_value=10000, max_value=300000, step=10000, value=st.session_state.autorefresh)
st.session_state.autorefresh = refresh_ms
st.sidebar.write(f"Dernière actualisation : {now_paris().strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.sidebar.info("L'app se réactualise à chaque interaction. Modifiez un paramètre pour rafraîchir.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.title("📈 Paper Trading — Suivi des rendements & live")
    st.caption("Données Yahoo Finance via yfinance (peuvent être retardées).")

    st.subheader("Passer un ordre fictif")
    invest_amount = st.number_input("Montant à investir (€)", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
    alloc_mode = st.radio("Méthode d'allocation", ["Égalitaire", "Pondérations personnalisées"])

    if alloc_mode == "Égalitaire":
        weights = pd.Series(1.0 / len(selected), index=selected) if len(selected) else pd.Series(dtype=float)
    else:
        st.write("Entrez des pondérations (0–100). Elles seront normalisées si la somme ≠ 100.")
        custom = {}
        for t in selected:
            custom[t] = st.number_input(f"Poids {t} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"w_{t}")
        w = pd.Series(custom) / 100.0 if len(custom) else pd.Series(dtype=float)
        if w.sum() == 0 and len(selected):
            st.warning("La somme des poids est 0. Utilisation de l'égalitaire à la place.")
            w = pd.Series(1.0 / len(selected), index=selected)
        elif w.sum() > 0:
            w = w / w.sum()
        weights = w

    if st.button("📥 Placer l'ordre (prix marché)") and len(selected):
        last = latest_prices(selected)
        order = place_order(invest_amount, weights, last)
        st.session_state.orders.append(order)
        ts = datetime.fromisoformat(order['timestamp']).astimezone(PARIS).strftime('%Y-%m-%d %H:%M:%S %Z')
        st.success(f"Ordre placé à {ts}. Coût ≈ {order['cost']:.2f}.")

with col_right:
    st.subheader("🧾 Ordres & Positions")
    if len(st.session_state.orders) == 0:
        st.info("Aucun ordre pour l'instant. Placez un ordre pour créer un portefeuille.")
    else:
        orders_df = pd.DataFrame(st.session_state.orders)
        st.dataframe(orders_df[["timestamp", "amount", "cost"]])

        pos = portfolio_positions(st.session_state.orders).sort_index()
        avg = portfolio_cost_basis(st.session_state.orders).reindex(pos.index)
        last_px = latest_prices(pos.index.tolist())
        df_pos = pd.DataFrame({
            "qty": pos,
            "avg_cost": avg,
            "last": last_px.reindex(pos.index),
        })
        df_pos["Mkt Value"] = (df_pos["qty"] * df_pos["last"]).round(2)
        df_pos["PnL %"] = ((df_pos["last"] / df_pos["avg_cost"]) - 1.0) * 100.0
        st.dataframe(df_pos.round(3))

        hist = fetch_history(pos.index.tolist(), start_date)
        equity = compute_equity_curve(hist, pos)
        if not equity.empty:
            daily_ret, cum_ret = compute_returns(equity)

            st.metric("Valeur portefeuille (aujourd'hui)", f"{equity.iloc[-1]:,.2f}")
            st.metric("Perf. cumulée depuis la date choisie", f"{(cum_ret.iloc[-1]*100):.2f}%")

            fig1, ax1 = plt.subplots()
            ax1.plot(equity.index, equity.values)
            ax1.set_title("Valeur du portefeuille (close quotidien)")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Valeur")
            st.pyplot(fig1, clear_figure=True)

            fig2, ax2 = plt.subplots()
            ax2.plot(cum_ret.index, (cum_ret.values*100.0))
            ax2.set_title("Rendement cumulé (%)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("%")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("Pas encore d'historique (aucune position).")

st.divider()
if st.button("♻️ Réinitialiser la session (efface les ordres)"):
    st.session_state.orders = []
    st.experimental_rerun()

st.caption("⚠️ Simulateur éducatif (paper trading). Aucune exécution réelle. Données indicatives.")
