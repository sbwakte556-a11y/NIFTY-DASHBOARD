import os
from pathlib import Path
from datetime import datetime, time as dtime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ====== CONFIG ======
try:
    from config import DATA_DIR as CFG_DATA_DIR, STRIKE_STEP, NEAR_STRIKES, TIMEZONE, REFRESH_SECS
    DATA_DIR = Path(os.path.abspath(CFG_DATA_DIR))
except Exception:
    DATA_DIR = Path(os.path.abspath("data"))
    STRIKE_STEP = 50
    NEAR_STRIKES = 3
    TIMEZONE = "Asia/Kolkata"
    REFRESH_SECS = 180

# ====== UTILS ======
def parse_ts_from_filename(path: Path) -> datetime:
    stem = path.stem.replace("snap_", "")
    return datetime.strptime(stem, "%Y%m%d_%H%M%S")

def load_one_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_file = parse_ts_from_filename(path)
    ts = ts_file
    df["timestamp"] = ts
    spot_val = float(df["spot"].median()) if "spot" in df.columns and df["spot"].notna().any() else np.nan
    df["spot"] = spot_val
    return df

def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    keep = ["strike", "option_type", "oi", "ltp", "spot", "timestamp"]
    df = df_long[keep].copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce")
    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
    df = df.dropna(subset=["strike"])
    piv_oi = df.pivot_table(index=["timestamp","spot","strike"], columns="option_type", values="oi", aggfunc="sum")
    piv_ltp = df.pivot_table(index=["timestamp","spot","strike"], columns="option_type", values="ltp", aggfunc="mean")
    wide = pd.concat([
        piv_oi.rename(columns={"CE":"ce_oi","PE":"pe_oi"}),
        piv_ltp.rename(columns={"CE":"ce_ltp","PE":"pe_ltp"})
    ], axis=1).reset_index()
    for c in ["ce_oi","pe_oi","ce_ltp","pe_ltp"]:
        if c not in wide.columns: wide[c] = np.nan
    return wide.sort_values(["timestamp","strike"]).reset_index(drop=True)

def build_per_strike_history() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("snap_*.csv"))
    if not files: return pd.DataFrame()
    wides = [make_wide(load_one_snapshot(f)) for f in files]
    hist = pd.concat(wides, ignore_index=True)
    hist = hist.sort_values(["strike","timestamp"]).reset_index(drop=True)
    for col in ["ce_oi","pe_oi","ce_ltp","pe_ltp"]:
        hist[f"{col}_chg"] = hist.groupby("strike")[col].diff()
    return hist

def build_master_history(per_strike: pd.DataFrame) -> pd.DataFrame:
    if per_strike.empty: return pd.DataFrame()
    agg = per_strike.groupby("timestamp", as_index=False).agg(
        total_ce_oi=("ce_oi","sum"),
        total_pe_oi=("pe_oi","sum"),
        spot=("spot","median")
    )
    ce = agg["total_ce_oi"].replace(0,np.nan)
    agg["pcr"] = agg["total_pe_oi"]/ce
    return agg.sort_values("timestamp")

def infer_atm_strike(wide_latest: pd.DataFrame) -> float:
    spot = float(wide_latest["spot"].median()) if wide_latest["spot"].notna().any() else np.nan
    if np.isfinite(spot):
        diffs = (wide_latest["strike"]-spot).abs()
        return float(wide_latest.loc[diffs.idxmin(),"strike"])
    ssum = (wide_latest["ce_ltp"].fillna(0)+wide_latest["pe_ltp"].fillna(0))
    return float(wide_latest.loc[ssum.idxmin(),"strike"])

def tag_buildup(oi_chg, px_chg):
    if pd.isna(oi_chg) or pd.isna(px_chg): return "Neutral"
    if oi_chg > 0 and px_chg > 0: return "Long Buildup"
    if oi_chg > 0 and px_chg < 0: return "Short Buildup"
    if oi_chg < 0 and px_chg > 0: return "Short Covering"
    if oi_chg < 0 and px_chg < 0: return "Long Unwinding"
    return "Neutral"

def buildup_color(val):
    mapping = {
        "Long Buildup": "background-color: #006400; color: white",
        "Long Unwinding": "background-color: yellow; color: black",
        "Short Buildup": "background-color: red; color: white",
        "Short Covering": "background-color: darkblue; color: white"
    }
    return mapping.get(val, "")

def market_sentiment_from_changes(per_strike: pd.DataFrame, ts_last: datetime) -> str:
    last_rows = per_strike[per_strike["timestamp"] == ts_last]
    ce = last_rows["ce_oi_chg"].sum(skipna=True)
    pe = last_rows["pe_oi_chg"].sum(skipna=True)
    if pe > ce and pe > 0: return "🐂 Bullish"
    if ce > pe and ce > 0: return "🐻 Bearish"
    return "⚖️ Neutral"

def buyer_seller_strength(per_strike: pd.DataFrame, ts_last: datetime) -> tuple:
    last_rows = per_strike[per_strike["timestamp"] == ts_last]
    ce = last_rows["ce_oi_chg"].sum(skipna=True)
    pe = last_rows["pe_oi_chg"].sum(skipna=True)
    total = abs(ce)+abs(pe)
    if total == 0: return 50,50
    buyers = round((pe/total)*100,2)
    sellers = round((ce/total)*100,2)
    return buyers, sellers

def within_market_hours(ts: pd.Timestamp) -> bool:
    lt = ts.to_pydatetime().time()
    return dtime(9,15) <= lt <= dtime(15,30)

# ====== STREAMLIT ======
st.set_page_config(page_title="Nifty Option Dashboard", layout="wide")
st.title("📊 Nifty Option Chain Dashboard")

files = sorted(DATA_DIR.glob("snap_*.csv"))
if not files:
    st.error(f"No snapshots found in {DATA_DIR}. Run nse_fetch.py first.")
    st.stop()

per_strike = build_per_strike_history()
master = build_master_history(per_strike)
latest_ts = per_strike["timestamp"].max()
latest_df = per_strike[per_strike["timestamp"]==latest_ts].copy()

atm = infer_atm_strike(latest_df)
pcr_latest = float(master.loc[master["timestamp"]==latest_ts,"pcr"].iloc[0]) if not master.empty else np.nan
spot_latest = float(latest_df["spot"].median()) if latest_df["spot"].notna().any() else np.nan

c1,c2,c3,c4 = st.columns(4)
c1.metric("Snapshot", latest_ts.strftime("%H:%M:%S"))
c2.metric("Spot", f"{spot_latest:,.2f}")
c3.metric("ATM", f"{atm:.0f}")
c4.metric("PCR", f"{pcr_latest:.2f}")

st.info(f"📌 Market Sentiment: {market_sentiment_from_changes(per_strike, latest_ts)}")

# Index Meter
buyers,sellers = buyer_seller_strength(per_strike, latest_ts)
fig_meter = go.Figure(go.Indicator(
    mode="gauge+number",
    value=buyers,
    gauge={"axis":{"range":[0,100]},"bar":{"color":"green"},
           "steps":[{"range":[0,50],"color":"red"},{"range":[50,100],"color":"green"}]},
    title={"text":"Buyer Strength (%)"}
))
st.plotly_chart(fig_meter, use_container_width=True)

# Top Strikes
st.subheader("🏆 Top Strikes — OI & Price")
top_n = st.slider("Top N", 5,15,10)
t1,t2 = st.columns(2)
with t1:
    st.write("Top CE OI")
    st.dataframe(latest_df.nlargest(top_n,"ce_oi")[["strike","ce_oi","ce_oi_chg"]])
    st.write("Top CE Price")
    st.dataframe(latest_df.nlargest(top_n,"ce_ltp")[["strike","ce_ltp","ce_ltp_chg"]])
with t2:
    st.write("Top PE OI")
    st.dataframe(latest_df.nlargest(top_n,"pe_oi")[["strike","pe_oi","pe_oi_chg"]])
    st.write("Top PE Price")
    st.dataframe(latest_df.nlargest(top_n,"pe_ltp")[["strike","pe_ltp","pe_ltp_chg"]])

# Buildup Table
st.subheader("🧮 Buildup Table")
buildup_df = latest_df[["strike","ce_oi_chg","ce_ltp_chg","pe_oi_chg","pe_ltp_chg"]].copy()
buildup_df["CE Buildup"] = [tag_buildup(o,p) for o,p in zip(buildup_df["ce_oi_chg"], buildup_df["ce_ltp_chg"])]
buildup_df["PE Buildup"] = [tag_buildup(o,p) for o,p in zip(buildup_df["pe_oi_chg"], buildup_df["pe_ltp_chg"])]
st.dataframe(buildup_df.style.applymap(buildup_color, subset=["CE Buildup","PE Buildup"]))

# OI Distribution
st.subheader("📊 OI Distribution")
fig_dist = go.Figure()
fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=latest_df["ce_oi"], name="CE OI"))
fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=-latest_df["pe_oi"], name="PE OI"))
st.plotly_chart(fig_dist, use_container_width=True)

# Straddle
st.subheader("🎯 Straddle (CE+PE)")
latest_df["straddle"] = latest_df["ce_ltp"].fillna(0)+latest_df["pe_ltp"].fillna(0)
fig_straddle = go.Figure()
fig_straddle.add_trace(go.Scatter(x=latest_df["strike"], y=latest_df["straddle"], mode="lines+markers"))
fig_straddle.add_vline(x=atm,line_dash="dash",annotation_text=f"ATM {atm:.0f}")
if np.isfinite(spot_latest):
    fig_straddle.add_vline(x=spot_latest,line_dash="dot",annotation_text=f"Spot {spot_latest:.0f}")
st.plotly_chart(fig_straddle, use_container_width=True)

# Trending OI
st.subheader("⏱️ Trending OI (ATM ± Strikes)")
sel_date = st.date_input("Trading Day", latest_ts.date())
day_hist = per_strike[(per_strike["timestamp"].dt.date==sel_date) & per_strike["timestamp"].apply(within_market_hours)]
if not day_hist.empty:
    first_ts = day_hist["timestamp"].min()
    base_atm = infer_atm_strike(day_hist[day_hist["timestamp"]==first_ts])
    strikes_window = [base_atm+i*STRIKE_STEP for i in range(-NEAR_STRIKES,NEAR_STRIKES+1)]
    day_win = day_hist[day_hist["strike"].isin(strikes_window)]
    tot_by_ts = day_win.groupby("timestamp",as_index=False).agg(ce_oi=("ce_oi","sum"),pe_oi=("pe_oi","sum"))
    fig_day = go.Figure()
    fig_day.add_trace(go.Scatter(x=tot_by_ts["timestamp"], y=tot_by_ts["ce_oi"], name="CE OI"))
    fig_day.add_trace(go.Scatter(x=tot_by_ts["timestamp"], y=tot_by_ts["pe_oi"], name="PE OI"))
    st.plotly_chart(fig_day, use_container_width=True)
# ====== Seller Shifting ======
st.subheader("🧭 Seller Shifting — Support/Resistance")

timestamps_sorted = sorted(per_strike["timestamp"].unique())
if len(timestamps_sorted) >= 2:
    ts_prev, ts_last = timestamps_sorted[-2], timestamps_sorted[-1]
    prev = per_strike[per_strike["timestamp"] == ts_prev]
    last = per_strike[per_strike["timestamp"] == ts_last]

    def top_strike(df, col):
        sub = df[["strike", col]].dropna()
        if sub.empty: return None
        return float(sub.loc[sub[col].idxmax(), "strike"])

    prev_ce = top_strike(prev, "ce_oi")
    last_ce = top_strike(last, "ce_oi")
    prev_pe = top_strike(prev, "pe_oi")
    last_pe = top_strike(last, "pe_oi")

    st.markdown(f"**Call OI Top (Resistance):** {prev_ce:.0f} ➝ {last_ce:.0f}" if prev_ce and last_ce else "—")
    st.markdown(f"**Put OI Top (Support):** {prev_pe:.0f} ➝ {last_pe:.0f}" if prev_pe and last_pe else "—")

    # Expert-style 2-line analysis
    if last_pe and last_ce:
        if last_pe > prev_pe and last_ce > prev_ce:
            st.success("⚡ Both support and resistance shifting up → market bias bullish.")
        elif last_pe < prev_pe and last_ce < prev_ce:
            st.error("⚠️ Both support and resistance shifting down → market bias bearish.")
        else:
            st.info("🔄 Mixed shift in support/resistance → rangebound/volatile setup.")
else:
    st.warning("Not enough snapshots to detect seller shifting.")
# ====== Seller Shifting — Support/Resistance + Big Add/Drop ======
st.subheader("🧭 Seller Shifting — Support/Resistance")

timestamps_sorted = sorted(per_strike["timestamp"].unique())
if len(timestamps_sorted) >= 2:
    ts_prev, ts_last = timestamps_sorted[-2], timestamps_sorted[-1]
    prev = per_strike[per_strike["timestamp"] == ts_prev]
    last = per_strike[per_strike["timestamp"] == ts_last]

    def top_strike(df, col):
        sub = df[["strike", col]].dropna()
        if sub.empty: return None
        return float(sub.loc[sub[col].idxmax(), "strike"])

    prev_ce = top_strike(prev, "ce_oi"); last_ce = top_strike(last, "ce_oi")
    prev_pe = top_strike(prev, "pe_oi"); last_pe = top_strike(last, "pe_oi")

    st.markdown(f"**Call OI Top (Resistance):** {prev_ce:.0f} ➝ {last_ce:.0f}" if prev_ce and last_ce else "—")
    st.markdown(f"**Put OI Top (Support):** {prev_pe:.0f} ➝ {last_pe:.0f}" if prev_pe and last_pe else "—")

    # Biggest changes (sellers’ activity)
    ce_big_add  = last.loc[last["ce_oi_chg"].idxmax()][["strike","ce_oi_chg"]] if last["ce_oi_chg"].notna().any() else None
    ce_big_drop = last.loc[last["ce_oi_chg"].idxmin()][["strike","ce_oi_chg"]] if last["ce_oi_chg"].notna().any() else None
    pe_big_add  = last.loc[last["pe_oi_chg"].idxmax()][["strike","pe_oi_chg"]] if last["pe_oi_chg"].notna().any() else None
    pe_big_drop = last.loc[last["pe_oi_chg"].idxmin()][["strike","pe_oi_chg"]] if last["pe_oi_chg"].notna().any() else None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Calls (CE)**")
        st.write(f"Big Add: {('—' if ce_big_add is None else f'{ce_big_add.strike:.0f} (+{ce_big_add.ce_oi_chg:,.0f})')}")
        st.write(f"Big Drop: {('—' if ce_big_drop is None else f'{ce_big_drop.strike:.0f} ({ce_big_drop.ce_oi_chg:,.0f})')}")

    with c2:
        st.markdown("**Puts (PE)**")
        st.write(f"Big Add: {('—' if pe_big_add is None else f'{pe_big_add.strike:.0f} (+{pe_big_add.pe_oi_chg:,.0f})')}")
        st.write(f"Big Drop: {('—' if pe_big_drop is None else f'{pe_big_drop.strike:.0f} ({pe_big_drop.pe_oi_chg:,.0f})')}")

    # 2-line expert analysis
    if last_pe and last_ce:
        if last_pe > prev_pe and last_ce > prev_ce:
            st.success("⚡ Support & Resistance both moving UP → bullish setup.")
        elif last_pe < prev_pe and last_ce < prev_ce:
            st.error("⚠️ Support & Resistance both moving DOWN → bearish setup.")
        else:
            st.info("🔄 Mixed shift → rangebound/volatile setup.")
else:
    st.warning("Not enough snapshots to detect seller shifting.")
# ====== CE vs PE Premium Crossover around ATM ======
st.subheader("📈 CE–PE Premium Crossover (ATM ± N Strikes)")

# take ATM ± window
atm = infer_atm_strike(latest_df)
strikes_window = [atm + i*STRIKE_STEP for i in range(-NEAR_STRIKES, NEAR_STRIKES+1)]
cross_df = latest_df[latest_df["strike"].isin(strikes_window)].copy()

if cross_df.empty:
    st.warning("No strikes found in ATM window.")
else:
    cross_df["premium_diff"] = cross_df["ce_ltp"].fillna(0) - cross_df["pe_ltp"].fillna(0)

    fig_cross = go.Figure()
    fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["ce_ltp"], mode="lines+markers", name="CE Premium"))
    fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["pe_ltp"], mode="lines+markers", name="PE Premium"))
    fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["premium_diff"], mode="lines+markers", name="CE-PE Diff"))

    fig_cross.update_layout(title=f"CE vs PE Premium Crossover (ATM {atm:.0f} ± {NEAR_STRIKES})",
                            xaxis_title="Strike", yaxis_title="Premium")
    st.plotly_chart(fig_cross, use_container_width=True)

    # Expert 2-line summary
    max_strike = cross_df.loc[cross_df["premium_diff"].idxmax(), "strike"]
    min_strike = cross_df.loc[cross_df["premium_diff"].idxmin(), "strike"]
    st.info(f"➡️ CE stronger vs PE near {max_strike:.0f}, while PE premium dominates near {min_strike:.0f}.")

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Filter short buildup near ATM (±250 points)
short_ce_near_atm = latest_df[
    (latest_df.apply(lambda x: tag_buildup(x["ce_oi_chg"], x["ce_ltp_chg"])=="Short Buildup", axis=1)) &
    (latest_df["strike"].between(atm-250, atm+250))
]

if not short_ce_near_atm.empty:
    st.subheader("⚠️ Short Buildup Near ATM")
    short_table = short_ce_near_atm[["strike","ce_oi","ce_oi_chg","ce_ltp","ce_ltp_chg"]].copy()
    # Add Buildup column
    short_table["Buildup"] = [tag_buildup(o,p) for o,p in zip(short_table["ce_oi_chg"], short_table["ce_ltp_chg"])]

    # AgGrid JS-based coloring
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.value == "Short Buildup") {return {color: 'white', backgroundColor: 'red'};}
        return {color: 'black', backgroundColor: 'white'};
    }
    """)
    gb = GridOptionsBuilder.from_dataframe(short_table)
    gb.configure_columns(["Buildup"], cellStyle=cell_style_jscode)
    gridOptions = gb.build()
    AgGrid(short_table, gridOptions=gridOptions, fit_columns_on_grid_load=True)
else:
    st.info("No Short Buildup detected near ATM")


# Raw Snapshot
with st.expander("🔎 Raw Latest Snapshot"):
    st.dataframe(latest_df.sort_values("strike"))
