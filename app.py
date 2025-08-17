# app.py
"""
NIFTY Live Option Chain — Colorful Dashboard (no Excel)
- Fetches NSE option-chain for NIFTY
- Computes 1-interval changes using session_state.prev_snapshot
- Shows Market Sentiment, Near-ATM analytics and Trending OI per selected strike (table + charts)
- Auto-refresh via streamlit_autorefresh (interval configurable in sidebar)
"""

import time
import io
from typing import Optional
import requests
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# ------------------------
# CONFIG / DEFAULTS
# ------------------------
st.set_page_config(layout="wide", page_title="NIFTY Options Live — Color", page_icon="📈")

SYMBOL = "NIFTY"
TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(TIMEZONE)
DEFAULT_REFRESH_SECS = 180  # 3 minutes
DEFAULT_NEAR_STRIKES = 3
STRIKE_STEP = 50

# ------------------------
# HELPERS
# ------------------------
def make_session():
    s = requests.Session()
    s.headers.update({
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "accept-language": "en-US,en;q=0.9",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "referer": "https://www.nseindia.com/",
        "connection": "keep-alive",
    })
    try:
        s.get("https://www.nseindia.com", timeout=8)
    except Exception:
        pass
    return s

def fetch_option_chain(symbol: str = SYMBOL, tries: int = 5, backoff: float = 1.5) -> pd.DataFrame:
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    s = make_session()
    headers = {"accept": "application/json, text/plain, */*", "referer": "https://www.nseindia.com/option-chain"}
    for attempt in range(tries):
        try:
            r = s.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                data = r.json()
                break
            time.sleep(backoff * (attempt + 1))
        except Exception:
            time.sleep(backoff * (attempt + 1))
    else:
        return pd.DataFrame()

    records = data.get("records", {})
    rows = []
    ts = dt.datetime.now(IST)
    spot = records.get("underlyingValue") or 0.0
    for item in records.get("data", []):
        strike = item.get("strikePrice")
        if strike is None:
            continue
        for side in ("CE", "PE"):
            leg = item.get(side)
            if not isinstance(leg, dict):
                continue
            rows.append({
                "symbol": symbol,
                "strike": int(strike),
                "option_type": side,
                "ltp": float(leg.get("lastPrice") or 0.0),
                "oi": float(leg.get("openInterest") or 0.0),
                "volume": float(leg.get("totalTradedVolume") or 0.0),
                "iv": float(leg.get("impliedVolatility") or 0.0),
                "vwap": np.nan,
                "ts": ts,
                "spot": float(spot or 0.0),
            })
    df = pd.DataFrame(rows)
    return df

def nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
    return int(round(price / step) * step)

def classify_buildup(oi_change: float, ltp_change: float) -> str:
    if oi_change > 0 and ltp_change > 0:
        return "Long Buildup"
    if oi_change > 0 and ltp_change < 0:
        return "Short Buildup"
    if oi_change < 0 and ltp_change < 0:
        return "Long Unwinding"
    if oi_change < 0 and ltp_change > 0:
        return "Short Covering"
    return "Neutral"

def enrich_with_prev(curr: pd.DataFrame, prev: Optional[pd.DataFrame]) -> pd.DataFrame:
    if curr.empty:
        return curr
    df = curr.copy()
    if prev is None or prev.empty:
        df["prev_ltp"] = df["ltp"]
        df["prev_oi"] = df["oi"]
    else:
        m = prev[["symbol","strike","option_type","ltp","oi"]].rename(columns={"ltp":"prev_ltp","oi":"prev_oi"})
        df = df.merge(m, on=["symbol","strike","option_type"], how="left")
        df["prev_ltp"] = df["prev_ltp"].fillna(df["ltp"])
        df["prev_oi"] = df["prev_oi"].fillna(df["oi"])
    df["oi_chg"] = df["oi"] - df["prev_oi"]
    df["ltp_chg"] = df["ltp"] - df["prev_ltp"]
    df["oi_chg_pct"] = np.where(df["prev_oi"]>0, 100*df["oi_chg"]/df["prev_oi"], 0.0)
    df["ltp_chg_pct"] = np.where(df["prev_ltp"]>0, 100*df["ltp_chg"]/df["prev_ltp"], 0.0)
    df["buildup"] = [classify_buildup(o, p) for o, p in zip(df["oi_chg"], df["ltp_chg"])]
    df["above_vwap"] = df["ltp"] > df["vwap"]
    return df

def select_near_atm(df: pd.DataFrame, spot: float, n: int = DEFAULT_NEAR_STRIKES) -> pd.DataFrame:
    if df.empty:
        return df
    atm = nearest_strike(spot)
    lo, hi = atm - n*STRIKE_STEP, atm + n*STRIKE_STEP
    return df[(df["strike"]>=lo) & (df["strike"]<=hi)].copy()

def compute_crossover(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    if df.empty:
        return pd.DataFrame(out)
    for (symbol, strike), g in df.groupby(["symbol","strike"]):
        ce = g[g.option_type=="CE"]["ltp"].values
        pe = g[g.option_type=="PE"]["ltp"].values
        if ce.size and pe.size:
            ce_val, pe_val = float(ce[0]), float(pe[0])
            out.append({
                "symbol": symbol,
                "strike": int(strike),
                "ce_gt_pe": bool(ce_val > pe_val),
                "pe_gt_ce": bool(pe_val > ce_val),
                "diff_pct": float((ce_val - pe_val) / max(1e-6, pe_val) * 100),
            })
    return pd.DataFrame(out)

# ------------------------
# SIDEBAR / SETTINGS
# ------------------------
st.sidebar.title("Settings")
refresh_secs = st.sidebar.number_input("Auto-refresh (seconds)", min_value=30, max_value=900, value=DEFAULT_REFRESH_SECS, step=30)
near_strikes = st.sidebar.slider("Strikes near ATM (±)", 1, 6, DEFAULT_NEAR_STRIKES)
oi_alert_pct = st.sidebar.slider("Exceptional OI% threshold", 5, 500, 50)
st.sidebar.markdown("---")
st.sidebar.markdown("App fetches live from NSE. If NSE blocks, please wait a bit and refresh.")

# trigger client-side refresh
st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")

# ------------------------
# FETCH DATA
# ------------------------
st.title("📈 NIFTY Options Live — Colorful Dashboard")
status = st.empty()

if "prev_snapshot" not in st.session_state:
    st.session_state.prev_snapshot = None

try:
    status.info("Fetching option-chain from NSE...")
    curr = fetch_option_chain(SYMBOL)
    if curr.empty:
        status.error("Failed to fetch data from NSE (empty). Try again shortly.")
        st.stop()
    status.success("Fetched live option chain.")
except Exception as e:
    status.error(f"Fetch failed: {e}")
    st.stop()

# Enrich with prev snapshot (1 interval change)
prev = st.session_state.prev_snapshot
df_en = enrich_with_prev(curr, prev)
st.session_state.prev_snapshot = curr[["symbol","strike","option_type","ltp","oi"]].copy()

# Header — metrics + colourful sentiment pill
spot = float(df_en["spot"].dropna().iloc[0]) if "spot" in df_en.columns and not df_en["spot"].dropna().empty else 0.0
snapshot_time = pd.to_datetime(df_en["ts"].iloc[0]) if "ts" in df_en.columns else dt.datetime.now(IST)

# compute buyer/seller strength proxy
ce = df_en[df_en.option_type == "CE"].copy()
pe = df_en[df_en.option_type == "PE"].copy()
ce["score"] = ce["volume"].fillna(0) * ce["ltp_chg"].abs().fillna(0)
pe["score"] = pe["volume"].fillna(0) * pe["ltp_chg"].abs().fillna(0)
ce_strength = float(ce["score"].sum())
pe_strength = float(pe["score"].sum())
total_strength = (ce_strength + pe_strength) or 1
buyer_pct = 100 * ce_strength / total_strength
seller_pct = 100 * pe_strength / total_strength

# directional counts for sentiment
up_count = ((df_en["buildup"] == "Long Buildup") | (df_en["buildup"] == "Short Covering")).sum()
down_count = ((df_en["buildup"] == "Short Buildup") | (df_en["buildup"] == "Long Unwinding")).sum()

# sentiment score: weighted mix of buyer_pct and directional counts
sent_score = 0.6 * (buyer_pct - seller_pct) + 0.4 * (up_count - down_count)
# map to label
if sent_score > 15:
    sentiment_label = "Bullish"
    sentiment_color = "#1b5e20"  # green
elif sent_score < -15:
    sentiment_label = "Bearish"
    sentiment_color = "#b71c1c"  # red
else:
    sentiment_label = "Neutral"
    sentiment_color = "#263238"  # grey

# header layout
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
with c1:
    st.markdown(f"**Spot (approx)**  \n:large_blue_circle: **{spot:.2f}**")
    st.markdown(f"**Snapshot**  \n{snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}")
with c2:
    st.metric("Buyer % (CE proxy)", f"{buyer_pct:.1f}%")
with c3:
    st.metric("Seller % (PE proxy)", f"{seller_pct:.1f}%")
with c4:
    # coloured sentiment pill
    st.markdown(f"""<div style="padding:10px;border-radius:10px;background:{sentiment_color};color:white;text-align:center">
                    <strong>Market Sentiment</strong><br><span style="font-size:20px">{sentiment_label}</span>
                    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------
# NEAR-ATM VIEW
# ------------------------
near = select_near_atm(df_en, spot, n=near_strikes)
st.subheader(f"Strikes around ATM (±{near_strikes * STRIKE_STEP} points) — showing {len(near)//2} strikes")

if near.empty:
    st.warning("No near-ATM data available.")
else:
    show_cols = ["strike","option_type","ltp","iv","oi","oi_chg_pct","ltp_chg_pct","buildup"]
    st.dataframe(near.sort_values(["strike","option_type"])[show_cols], use_container_width=True)

    cross = compute_crossover(near)
    st.markdown("**CE vs PE crossover (near ATM)**")
    st.dataframe(cross.sort_values("strike"), use_container_width=True)

# ------------------------
# TRENDING OI (user-selectable strike)
# ------------------------
st.markdown("---")
st.subheader("🔎 Trending OI — pick a strike to inspect")

# strike choices come from available strikes in current snapshot
available_strikes = sorted(df_en["strike"].unique()) if not df_en.empty else []
sel_str = st.selectbox("Select strike", options=available_strikes, index=len(available_strikes)//2 if available_strikes else 0)

if sel_str:
    # filter df_en for this strike
    s = df_en[(df_en.strike == sel_str)]
    ce_row = s[s.option_type == "CE"].squeeze() if not s[s.option_type=="CE"].empty else None
    pe_row = s[s.option_type == "PE"].squeeze() if not s[s.option_type=="PE"].empty else None

    # Prepare summary numbers (safely)
    def safe_val(row, col):
        return float(row[col]) if (row is not None and col in row and pd.notna(row[col])) else 0.0

    ce_oi = safe_val(ce_row, "oi")
    pe_oi = safe_val(pe_row, "oi")
    ce_prev_oi = safe_val(ce_row, "prev_oi")
    pe_prev_oi = safe_val(pe_row, "prev_oi")
    ce_oi_chg = ce_oi - ce_prev_oi
    pe_oi_chg = pe_oi - pe_prev_oi
    ce_oi_chg_pct = (100 * ce_oi_chg / ce_prev_oi) if ce_prev_oi>0 else (100 if ce_oi_chg>0 else 0)
    pe_oi_chg_pct = (100 * pe_oi_chg / pe_prev_oi) if pe_prev_oi>0 else (100 if pe_oi_chg>0 else 0)

    ce_ltp = safe_val(ce_row, "ltp")
    pe_ltp = safe_val(pe_row, "ltp")
    ce_prev_ltp = safe_val(ce_row, "prev_ltp")
    pe_prev_ltp = safe_val(pe_row, "prev_ltp")
    ce_ltp_chg = ce_ltp - ce_prev_ltp
    pe_ltp_chg = pe_ltp - pe_prev_ltp
    ce_ltp_chg_pct = (100 * ce_ltp_chg / ce_prev_ltp) if ce_prev_ltp>0 else 0
    pe_ltp_chg_pct = (100 * pe_ltp_chg / pe_prev_ltp) if pe_prev_ltp>0 else 0

    # display summary cards
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#e8f5e9'><strong>CE OI</strong><br><span style='font-size:20px'>{int(ce_oi):,}</span><br><small>Δ {int(ce_oi_chg):+,}</small></div>", unsafe_allow_html=True)
    with a2:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#ffebee'><strong>PE OI</strong><br><span style='font-size:20px'>{int(pe_oi):,}</span><br><small>Δ {int(pe_oi_chg):+,}</small></div>", unsafe_allow_html=True)
    with a3:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#e3f2fd'><strong>CE LTP</strong><br><span style='font-size:20px'>{ce_ltp:.2f}</span><br><small>Δ {ce_ltp_chg:+.2f} ({ce_ltp_chg_pct:+.1f}%)</small></div>", unsafe_allow_html=True)
    with a4:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#fff8e1'><strong>PE LTP</strong><br><span style='font-size:20px'>{pe_ltp:.2f}</span><br><small>Δ {pe_ltp_chg:+.2f} ({pe_ltp_chg_pct:+.1f}%)</small></div>", unsafe_allow_html=True)

    # tabbed view: table + charts
    tab1, tab2 = st.tabs(["Summary Table","Charts"])
    with tab1:
        # tabular summary
        table = {
            "side": ["CE","PE"],
            "oi": [ce_oi, pe_oi],
            "oi_prev": [ce_prev_oi, pe_prev_oi],
            "oi_chg": [ce_oi_chg, pe_oi_chg],
            "oi_chg_pct": [round(ce_oi_chg_pct,2), round(pe_oi_chg_pct,2)],
            "ltp": [ce_ltp, pe_ltp],
            "ltp_prev": [ce_prev_ltp, pe_prev_ltp],
            "ltp_chg": [round(ce_ltp_chg,4), round(pe_ltp_chg,4)],
            "ltp_chg_pct": [round(ce_ltp_chg_pct,2), round(pe_ltp_chg_pct,2)],
            "buildup": [safe_val(ce_row, "buildup") if ce_row is not None else "NA", safe_val(pe_row, "buildup") if pe_row is not None else "NA"]
        }
        df_table = pd.DataFrame(table)
        st.dataframe(df_table, use_container_width=True)

    with tab2:
        # Chart 1: OI comparison
        fig1 = go.Figure(data=[
            go.Bar(name="CE OI", x=["CE","PE"], y=[ce_oi, pe_oi], marker_color=["#1b5e20","#b71c1c"]),
        ])
        fig1.update_layout(title_text=f"Total OI at strike {sel_str}", height=320, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: OI change %
        fig2 = go.Figure(data=[
            go.Bar(name="OI change %", x=["CE","PE"], y=[ce_oi_chg_pct, pe_oi_chg_pct], marker_color=["#4caf50","#ff7043"]),
        ])
        fig2.update_layout(title_text="OI change % vs previous snapshot", height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: LTP & LTP change
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="LTP", x=["CE","PE"], y=[ce_ltp, pe_ltp], marker_color=["#1976d2","#fbc02d"]))
        fig3.update_layout(title_text="Premium (LTP) — CE vs PE", height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------
# TOP MOVERS PANEL
# ------------------------
st.markdown("---")
st.subheader("Top movers (last interval) — by LTP % change")
if "ltp_chg_pct" in df_en.columns:
    top = df_en.assign(pct=df_en["ltp_chg_pct"]).sort_values("pct", ascending=False).head(10)
    st.dataframe(top[["strike","option_type","ltp","ltp_chg_pct","oi","oi_chg_pct"]], use_container_width=True)
else:
    st.info("First snapshot — come back after next refresh.")

# footer
st.markdown("---")
st.caption("This dashboard fetches NSE option-chain live. Auto-refresh interval is set in Settings. Sentiment and metrics are heuristic (short-interval). Use with caution.")
