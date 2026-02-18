import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import yfinance as yf

# =============================================================================
# LOAD TRANSACTION HISTORY & STATIC INFO FROM EXCEL
# Expected sheets:
#   "Transactions"  – columns: Date | Ticker | Action | Shares
#                    (Ticker is already in Yahoo Finance format — no mapping needed)
#   "Holdings_Info" – columns: Ticker | Name | Target_price | Currency |
#                              Sector | thesis | WACC | CF_1..CF_5
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """
    Reads a single Excel sheet that contains BOTH transaction rows AND per-ticker
    static info columns side-by-side.

    Required columns (transaction data — one row per trade):
        Date | Ticker | Action | Shares

    Optional extra columns (static info — only the first row per ticker is used):
        Name | Target_price | Currency | Sector | thesis | WACC | CF_1..CF_5

    If the static-info columns are absent, sensible defaults are used so the
    dashboard still loads; just fill them in to unlock all features.
    """
    EXCEL_PATH = "portfolio_holdings.xlsx"
    xl         = pd.ExcelFile(EXCEL_PATH)
    sheet_names = xl.sheet_names
    print(f"Sheets found in Excel: {sheet_names}")

    # ── Read the sheet that contains Buy/Sell data ────────────────────────────
    raw = None
    used_sheet = None
    for sname in sheet_names:
        candidate = pd.read_excel(xl, sheet_name=sname)
        candidate.columns = [str(c).strip() for c in candidate.columns]
        for col in candidate.columns:
            vals = candidate[col].dropna().astype(str).str.strip().str.lower().unique()
            if any(v in ["buy", "sell"] for v in vals):
                raw = candidate
                used_sheet = sname
                print(f"Found  data in sheet: '{sname}'")
                break
        if raw is not None:
            break

    if raw is None:
        raise ValueError(
            f"Could not find any sheet with Buy/Sell data. "
            f"Sheets available: {sheet_names}"
        )

    raw.columns = [str(c).strip() for c in raw.columns]
    col_lower   = {c.lower(): c for c in raw.columns}

    # ── Identify the four core  columns ────────────────────────────
    date_col   = next((col_lower[k] for k in ["date", "trade date", "tradedate"]           if k in col_lower), raw.columns[0])
    ticker_col = next((col_lower[k] for k in ["ticker", "security", "symbol", "stock"]     if k in col_lower), raw.columns[1])
    action_col = next((col_lower[k] for k in ["action", "type", "side", "buy/sell"]        if k in col_lower), raw.columns[2])
    qty_col    = next((col_lower[k] for k in ["shares", "quantity", "qty", "units", "volume"] if k in col_lower), raw.columns[3])

    print(f"   columns detected → date='{date_col}' ticker='{ticker_col}' action='{action_col}' qty='{qty_col}'")

    # ── Build clean s DataFrame ────────────────────────────────────
    tx = raw[[date_col, ticker_col, action_col, qty_col]].copy()
    tx.columns = ["Date", "Ticker", "Action", "Quantity"]

    tx = tx.dropna(subset=["Date", "Ticker", "Action", "Quantity"])
    tx["Date"]     = pd.to_datetime(tx["Date"], errors="coerce")
    tx             = tx.dropna(subset=["Date"])
    tx["Ticker"]   = tx["Ticker"].astype(str).str.strip()
    tx["Action"]   = tx["Action"].astype(str).str.strip().str.capitalize()
    tx["Quantity"] = pd.to_numeric(
        tx["Quantity"].astype(str).str.replace(",", "").str.replace(" ", ""),
        errors="coerce"
    ).fillna(0).astype(int)
    tx = tx[tx["Quantity"] > 0]
    tx["YF_Ticker"] = tx["Ticker"]   # already in Yahoo Finance format

    # ── Build static info DataFrame (one row per unique ticker) ───────────────
    # Identify which columns are the static-info extras (anything beyond the 4 core ones)
    core_cols = {date_col, ticker_col, action_col, qty_col}
    info_extra_cols = [c for c in raw.columns if c not in core_cols]

    # Normalise column names for flexible lookup
    info_col_map = {c.lower(): c for c in info_extra_cols}

    # For each unique ticker take the FIRST non-null row that has static data
    unique_tickers = tx["Ticker"].unique()

    INFO_DEFAULTS = {
        "name":         lambda t: t,          # fallback: use ticker as name
        "target_price": lambda t: 0.0,
        "currency":     lambda t: "USD",
        "sector":       lambda t: "Unknown",
        "thesis":       lambda t: "",
        "wacc":         lambda t: "",
        "cf_1":         lambda t: "",
        "cf_2":         lambda t: "",
        "cf_3":         lambda t: "",
        "cf_4":         lambda t: "",
        "cf_5":         lambda t: "",
    }

    # Map flexible column name → standard key
    FIELD_CANDIDATES = {
        "name":         ["name", "company", "company name"],
        "target_price": ["target_price", "target price", "targetprice", "target", "tp", "price target", "pt", "target px", "targetpx"],
        "currency":     ["currency", "ccy", "cur"],
        "sector":       ["sector", "industry", "gics sector"],
        "thesis":       ["thesis", "investment thesis", "rationale"],
        "wacc":         ["wacc"],
        "cf_1":         ["cf_1", "cf1", "cashflow_1", "cashflow1"],
        "cf_2":         ["cf_2", "cf2", "cashflow_2", "cashflow2"],
        "cf_3":         ["cf_3", "cf3", "cashflow_3", "cashflow3"],
        "cf_4":         ["cf_4", "cf4", "cashflow_4", "cashflow4"],
        "cf_5":         ["cf_5", "cf5", "cashflow_5", "cashflow5"],
    }

    def resolve_col(field):
        """Return the actual column name from the sheet for a given standard field, or None."""
        for candidate in FIELD_CANDIDATES[field]:
            if candidate in info_col_map:
                return info_col_map[candidate]
        return None

    # ── Fetch currency from yfinance for each unique ticker ──────────────────
    def fetch_currency_from_yf(ticker):
        """Query yfinance for the trading currency of a ticker."""
        try:
            info = yf.Ticker(ticker).info
            # yfinance exposes 'currency' directly
            ccy = info.get("currency", None)
            if ccy and isinstance(ccy, str) and len(ccy) == 3:
                return ccy.upper()
        except Exception:
            pass
        return None   # caller will use "USD" as final fallback

    print("Fetching currencies from yfinance…")
    yf_currencies = {}
    for ticker in unique_tickers:
        ccy = fetch_currency_from_yf(ticker)
        yf_currencies[ticker] = ccy if ccy else "USD"
        print(f"  {ticker}: {yf_currencies[ticker]}")

    # Override the currency default to use the yfinance-fetched value
    INFO_DEFAULTS["currency"] = lambda t: yf_currencies.get(t, "USD")

    # Print exactly which static columns were detected
    detected = {field: resolve_col(field) for field in FIELD_CANDIDATES}
    print(f"\nStatic columns detected in Excel: {detected}")

    info_records = {}
    for ticker in unique_tickers:
        # All rows for this ticker (Buy + Sell) — search all for static values
        ticker_rows = raw[raw[ticker_col].astype(str).str.strip() == ticker]
        rec = {}
        for field, default_fn in INFO_DEFAULTS.items():
            col = resolve_col(field)

            # Currency: always use yfinance value (most reliable)
            if field == "currency":
                rec[field] = yf_currencies.get(ticker, "USD")
                continue

            if col is not None:
                # Search ALL rows for this ticker for a non-null, non-empty value
                vals = ticker_rows[col].dropna()
                vals = vals[vals.astype(str).str.strip() != ""]
                rec[field] = vals.iloc[0] if len(vals) > 0 else default_fn(ticker)
            else:
                rec[field] = default_fn(ticker)

        info_records[ticker] = rec

    info_df = pd.DataFrame.from_dict(info_records, orient="index")
    info_df.index.name = "Ticker"

    # Print a summary so you can verify what was read
    print(f"\nSheet: '{used_sheet}' | s: {len(tx)} rows | Unique tickers: {len(info_df)}")
    print(f"{'Ticker':<20} {'Currency':<8} {'Sector':<20} {'Thesis':<30}")
    print("-" * 80)
    for t in sorted(info_df.index):
        row = info_df.loc[t]
        print(f"{t:<20} {str(row.get('currency','?')):<8} {str(row.get('sector','?')):<20} {str(row.get('thesis',''))[:28]}")

    return tx, info_df


def build_portfolio(tx: pd.DataFrame, info_df: pd.DataFrame) -> dict:
    """
    Compute current net positions from  history.

    info_df is indexed by ticker and has lowercase columns:
        name, target_price, currency, sector, thesis, wacc, cf_1..cf_5
    (built by load_data from the single Excel sheet)
    """
    portfolio = {}

    for yf_ticker, grp in tx.groupby("YF_Ticker"):
        if pd.isna(yf_ticker):
            continue

        buys  = grp[grp["Action"] == "Buy"]
        sells = grp[grp["Action"] == "Sell"]

        net_qty = int(buys["Quantity"].sum()) - int(sells["Quantity"].sum())
        if net_qty <= 0:
            continue   # position fully closed — skip

        # ── Look up static info row for this ticker ───────────────────────────
        if yf_ticker in info_df.index:
            row = info_df.loc[yf_ticker]
        else:
            # Should not happen since info_df is built from the same tx data,
            # but guard against edge cases gracefully
            st.warning(f"⚠️ No static info for **{yf_ticker}** – using defaults.")
            row = pd.Series({
                "name": yf_ticker, "target_price": 0.0, "currency": "USD",
                "sector": "Unknown", "thesis": "", "wacc": "",
                "cf_1": "", "cf_2": "", "cf_3": "", "cf_4": "", "cf_5": "",
            })

        # ── First buy date & VWAP purchase price ─────────────────────────────
        first_buy_date = buys["Date"].min()
        purchase_price = _vwap_purchase_price(yf_ticker, buys)

        # ── Safe value extraction helpers ─────────────────────────────────────
        def safe_float(val, default=0.0):
            try:    return float(val)
            except: return default

        def safe_str(val, default=""):
            return str(val) if pd.notna(val) else default

        portfolio[yf_ticker] = {
            "quantity":       net_qty,
            "name":           safe_str(row.get("name", yf_ticker), yf_ticker),
            "Target_price":   safe_float(row.get("target_price", 0.0)),
            "currency":       safe_str(row.get("currency", "USD"), "USD"),
            "sector":         safe_str(row.get("sector", "Unknown"), "Unknown"),
            "purchase_date":  first_buy_date.strftime("%Y-%m-%d"),
            "purchase_price": purchase_price,
            "thesis":         safe_str(row.get("thesis", "")),
            "WACC":           safe_str(row.get("wacc", "")),
            "CF_1":           safe_str(row.get("cf_1", "")),
            "CF_2":           safe_str(row.get("cf_2", "")),
            "CF_3":           safe_str(row.get("cf_3", "")),
            "CF_4":           safe_str(row.get("cf_4", "")),
            "CF_5":           safe_str(row.get("cf_5", "")),
            "_s":  grp.reset_index(drop=True),
        }

    return portfolio


def _vwap_purchase_price(ticker: str, buys: pd.DataFrame) -> float:
    """
    Weighted-average purchase price across all buy s.
    Fetches the closing price on each buy date from yfinance.
    Returns 0.0 on failure.
    """
    total_cost = 0.0
    total_qty  = 0

    for _, row in buys.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        qty      = int(row["Quantity"])
        price    = _fetch_close_on_date(ticker, row["Date"])
        if price is None:
            price = 0.0
        total_cost += price * qty
        total_qty  += qty

    return (total_cost / total_qty) if total_qty > 0 else 0.0


def _fetch_close_on_date(ticker: str, date: pd.Timestamp) -> float | None:
    """Return the closing price for *ticker* on *date* (or last available day)."""
    try:
        start = date - pd.Timedelta(days=7)
        end   = date + pd.Timedelta(days=1)
        data  = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        avail = close.index[close.index <= date]
        return float(close.loc[avail[-1]]) if len(avail) > 0 else None
    except Exception as e:
        print(f"Error fetching price for {ticker} on {date}: {e}")
        return None


# ── Load & build ──────────────────────────────────────────────────────────────
print("Loading  history…")
_tx, _info_df = load_data()

print("Building portfolio from s…")
portfolio_holdings = build_portfolio(_tx, _info_df)

print(f"Active positions: {list(portfolio_holdings.keys())}")

# Currency pairs for conversion to CHF (base currency)
# All known currency → CHF pairs. "CHF": None means no conversion needed.
# Any currency NOT listed here will be fetched dynamically from yfinance at runtime.
currency_pairs = {
    "USD": "USDCHF=X",
    "EUR": "EURCHF=X",
    "INR": "INRCHF=X",
    "HKD": "HKDCHF=X",
    "AUD": "AUDCHF=X",
    "CAD": "CADCHF=X",
    "GBP": "GBPCHF=X",
    "JPY": "JPYCHF=X",
    "CNY": "CNYCHF=X",
    "SEK": "SEKCHF=X",
    "NOK": "NOKCHF=X",
    "DKK": "DKKCHF=X",
    "SGD": "SGDCHF=X",
    "KRW": "KRWCHF=X",
    "BRL": "BRLCHF=X",
    "MXN": "MXNCHF=X",
    "CHF": None,   # base currency — no conversion needed
}

def get_fx_pair(currency: str) -> str | None:
    """Return the yfinance FX ticker for currency → CHF, or None if CHF."""
    if currency == "CHF":
        return None
    if currency in currency_pairs:
        return currency_pairs[currency]
    # Dynamically construct pair for any unlisted currency
    return f"{currency}CHF=X"

# Runtime cache for FX rate data (avoids re-downloading per page)
_fx_data_cache: dict = {}

def get_fx_data(currency: str, start, end) -> object:
    """Download and cache FX data for a currency over a date range."""
    pair = get_fx_pair(currency)
    if pair is None:
        return None   # CHF — rate is always 1.0
    cache_key = (pair, str(start), str(end))
    if cache_key not in _fx_data_cache:
        try:
            _fx_data_cache[cache_key] = yf.download(pair, start=start, end=end, progress=False)
        except Exception:
            _fx_data_cache[cache_key] = pd.DataFrame()
    return _fx_data_cache[cache_key]

# Fallback spot rates to CHF (used when yfinance download fails)
FALLBACK_FX_TO_CHF = {
    "USD": 0.888, "EUR": 0.940, "INR": 0.0104, "HKD": 0.114,
    "AUD": 0.570, "CAD": 0.650, "GBP": 1.120,  "JPY": 0.0059,
    "CNY": 0.122, "SGD": 0.660, "SEK": 0.083,  "NOK": 0.083,
    "DKK": 0.126, "KRW": 0.00065,
}
def get_fx_rate_for_date(currency: str, date, fx_data) -> float:
    """
    Get the CHF conversion rate for *currency* on *date* using pre-fetched fx_data.
    fx_data can be a DataFrame (from yf.download) or None (for CHF).
    Falls back to FALLBACK_FX_TO_CHF if data is unavailable.
    """
    if currency == "CHF":
        return 1.0
    if fx_data is None or (hasattr(fx_data, "empty") and fx_data.empty):
        return FALLBACK_FX_TO_CHF.get(currency, 1.0)
    try:
        close = fx_data["Close"].iloc[:, 0] if isinstance(fx_data["Close"], pd.DataFrame) else fx_data["Close"]
        avail = close.index[close.index <= date]
        if len(avail) > 0:
            return float(close.loc[avail[-1]])
        return float(close.iloc[0])
    except Exception:
        return FALLBACK_FX_TO_CHF.get(currency, 1.0)



# =============================================================================
# PAGE CONFIG & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #030C30; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] label { color: white !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] { background-color: transparent; }
    [data-testid="stSidebar"] [data-baseweb="radio"] > div { color: white !important; }
    .main { background-color: white; }
    div.stButton > button {
        height: 120px; white-space: pre-wrap;
        background-color: #f0f2f6; border: 2px solid #e0e2e6;
        color: #0F1D64; font-size: 16px; font-weight: 500;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0F1D64; color: white; border-color: #0F1D64;
        transform: translateY(-2px); box-shadow: 0 4px 12px rgba(15,29,100,0.3);
    }
    [data-testid="stSidebar"] div.stButton > button {
        height: auto; background-color: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.3); color: white;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.5);
        transform: none; box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("Navigation")

try:
    st.sidebar.image("HIC_Capital_Logo.png", use_container_width=True)
    st.sidebar.markdown("---")
except Exception:
    st.sidebar.markdown("### Portfolio Dashboard")
    st.sidebar.markdown("---")

if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.main_page = "Home"

if "main_page" not in st.session_state:
    st.session_state.main_page = "Home"
if st.sidebar.button("📋 s",    use_container_width=True): st.session_state.main_page = "Transactions"
    
st.sidebar.markdown("---")
st.sidebar.subheader("Sectors")

if st.sidebar.button("📱 TMT",             use_container_width=True): st.session_state.main_page = "TMT Sector"
if st.sidebar.button("🏦 FIG",             use_container_width=True): st.session_state.main_page = "FIG Sector"
if st.sidebar.button("🏭 Industrials",     use_container_width=True): st.session_state.main_page = "Industrials Sector"
if st.sidebar.button("⚡ PUI",             use_container_width=True): st.session_state.main_page = "PUI Sector"
if st.sidebar.button("🛒 Consumer Goods",  use_container_width=True): st.session_state.main_page = "Consumer Goods Sector"
if st.sidebar.button("🏥 Healthcare",      use_container_width=True): st.session_state.main_page = "Healthcare Sector"

main_page = st.session_state.main_page

# =============================================================================
# TRANSACTION HISTORY PAGE
# =============================================================================
if main_page == "Transactions":
    st.title("📋 Transaction History")

    # ── Fetch execution prices for every trade ────────────────────────────────
    @st.cache_data(show_spinner=False)
    def fetch_execution_prices(tx_df: pd.DataFrame) -> pd.Series:
        """
        For each row in tx_df, fetch the closing price on the trade date.
        Returns a Series of prices aligned to tx_df's index.
        """
        prices = []
        for _, row in tx_df.iterrows():
            ticker = row["Ticker"]
            date   = row["Date"]
            try:
                start = date - pd.Timedelta(days=7)
                end   = date + pd.Timedelta(days=1)
                data  = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if not data.empty:
                    close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
                    avail = close.index[close.index <= date]
                    prices.append(float(close.loc[avail[-1]]) if len(avail) > 0 else None)
                else:
                    prices.append(None)
            except Exception:
                prices.append(None)
        return pd.Series(prices, index=tx_df.index)

    # Show raw log with execution prices
    st.subheader("All Transactions")

    with st.spinner("Fetching execution prices from yfinance…"):
        exec_prices = fetch_execution_prices(_tx.reset_index(drop=True))

    display_tx = _tx.reset_index(drop=True).copy()
    display_tx["Exec Price"] = exec_prices.values

    # Pull currency per ticker from portfolio_holdings (or info_df fallback)
    def get_ticker_currency(ticker):
        if ticker in portfolio_holdings:
            return portfolio_holdings[ticker]["currency"]
        if ticker in _info_df.index:
            return str(_info_df.loc[ticker, "currency"])
        return ""

    display_tx["Currency"] = display_tx["Ticker"].apply(get_ticker_currency)

    # Gross value = price × shares
    display_tx["Gross Value"] = display_tx.apply(
        lambda r: r["Exec Price"] * r["Quantity"] if pd.notna(r["Exec Price"]) else None, axis=1
    )

    # Format for display
    display_tx["Date"]        = display_tx["Date"].dt.strftime("%Y-%m-%d")
    display_tx["Exec Price"]  = display_tx["Exec Price"].apply(
        lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
    )
    display_tx["Gross Value"] = display_tx["Gross Value"].apply(
        lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
    )

    st.dataframe(
        display_tx[["Date", "Ticker", "Action", "Quantity", "Currency", "Exec Price", "Gross Value"]],
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # Per-ticker position summary
    st.subheader("Net Position Summary")
    rows = []
    for yf_ticker, grp in _tx.groupby("YF_Ticker"):
        if pd.isna(yf_ticker):
            continue
        bought = int(grp[grp["Action"] == "Buy"]["Quantity"].sum())
        sold   = int(grp[grp["Action"] == "Sell"]["Quantity"].sum())
        net    = bought - sold
        rows.append({
            "YF Ticker": yf_ticker,
            "Bought": bought,
            "Sold":   sold,
            "Net Position": net,
            "Status": "🟢 Open" if net > 0 else "🔴 Closed",
        })
    summary_df = pd.DataFrame(rows).sort_values("YF Ticker")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Transaction timeline chart — bubble size = gross value, proportional across ALL bubbles
    st.subheader("Transaction Timeline")

    # We need exec prices to compute gross values for bubble sizing
    # Reuse the already-fetched exec_prices (numeric, before formatting)
    with st.spinner("Computing bubble sizes from trade values…"):
        raw_prices = fetch_execution_prices(_tx.reset_index(drop=True))

    bubble_tx = _tx.reset_index(drop=True).copy()
    bubble_tx["exec_price"] = raw_prices.values

    # Convert exec price to USD so all bubbles are on the same scale
    @st.cache_data(show_spinner=False)
    def get_spot_usd_rate(currency: str) -> float:
        """Fetch current spot rate currency → USD."""
        if currency == "USD":
            return 1.0
        try:
            pair = f"{currency}USD=X"
            d = yf.Ticker(pair).history(period="2d")
            if not d.empty:
                return float(d["Close"].iloc[-1])
        except Exception:
            pass
        # Fallback: invert CHF rates
        chf_rate = FALLBACK_FX_TO_CHF.get(currency, 1.0)
        usd_chf  = FALLBACK_FX_TO_CHF.get("USD", 0.888)
        return chf_rate / usd_chf if usd_chf else 1.0

    spot_usd = {t: get_spot_usd_rate(
        portfolio_holdings.get(t, {}).get("currency", "USD")
    ) for t in bubble_tx["Ticker"].unique()}

    bubble_tx["exec_price_usd"] = bubble_tx.apply(
        lambda r: r["exec_price"] * spot_usd.get(r["Ticker"], 1.0)
        if pd.notna(r["exec_price"]) else None, axis=1
    )
    bubble_tx["gross_value"] = bubble_tx.apply(
        lambda r: r["exec_price_usd"] * r["Quantity"]
        if pd.notna(r["exec_price_usd"]) else r["Quantity"],
        axis=1
    )

    # Global max across ALL rows (both Buy and Sell) so bubbles are comparable
    global_max_val = bubble_tx["gross_value"].max()
    # Scale: min bubble = 8, max bubble = 50
    bubble_tx["bubble_size"] = (bubble_tx["gross_value"] / global_max_val) * 42 + 8

    fig = go.Figure()
    colors = {"Buy": "#0F1D64", "Sell": "#FF6B6B"}
    for action in ["Buy", "Sell"]:
        subset = bubble_tx[bubble_tx["Action"] == action]
        fig.add_trace(go.Scatter(
            x=subset["Date"],
            y=subset["Ticker"],
            mode="markers",
            marker=dict(
                color=colors[action],
                size=subset["bubble_size"],
                opacity=0.8,
                sizemode="diameter",
            ),
            name=action,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Date: %{x}<br>"
                "Qty: %{customdata[0]:,}<br>"
                "Value (USD): $%{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
            customdata=subset[["Quantity","gross_value"]].values,
        ))
    fig.update_layout(
        title="Buy & Sell Events (bubble size = trade value in USD, proportional across all trades)",
        xaxis_title="Date", yaxis_title="Ticker",
        height=520, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# HOME PAGE
# =============================================================================
elif main_page == "Home":
    st.title("🏠 Portfolio Dashboard - Home")

    if "home_tab" not in st.session_state:
        st.session_state.home_tab = "Generic Summary"

    st.markdown("### Select Analysis Type")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Generic Summary\n\nKey metrics, performance vs MSCI World, portfolio treemap",
                     key="gen_summary", use_container_width=True):
            st.session_state.home_tab = "Generic Summary"
    with col2:
        if st.button("🏗️ Portfolio Structure\n\nSector, geographical, and asset distribution",
                     key="portfolio_struct", use_container_width=True):
            st.session_state.home_tab = "Portfolio Structure Analysis"
    with col3:
        if st.button("🔮 Forecast\n\nMonte Carlo, analyst targets, DCF analysis",
                     key="forecast", use_container_width=True):
            st.session_state.home_tab = "Forecast"

    st.markdown("---")
    home_tab = st.session_state.home_tab

    # -------------------------------------------------------------------------
    # HOME - Generic Summary
    # -------------------------------------------------------------------------
    if home_tab == "Generic Summary":
        st.header("📈 Generic Summary")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2025-11-06"),
                                       help="Select the start date for analysis")
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("today"),
                                     help="Select the end date for analysis")

        if st.button("📊 Generate Analysis", type="primary"):
            if start_date >= end_date:
                st.error("Start date must be before end date.")
            else:
                try:
                    with st.spinner("Fetching market data and exchange rates…"):
                        msci_world = yf.download("URTH", start=start_date, end=end_date, progress=False)

                        # Collect all currencies actually used in the portfolio
                        used_currencies = {info["currency"] for info in portfolio_holdings.values()}
                        exchange_rates = {}
                        for currency in used_currencies:
                            if currency == "CHF":
                                exchange_rates[currency] = None
                            else:
                                pair = get_fx_pair(currency)
                                if pair:
                                    try:
                                        exchange_rates[currency] = yf.download(pair, start=start_date, end=end_date, progress=False)
                                    except Exception:
                                        exchange_rates[currency] = pd.DataFrame()
                                else:
                                    exchange_rates[currency] = None

                        portfolio_data  = {}
                        initial_prices  = {}
                        current_prices_ = {}

                        for ticker, info in portfolio_holdings.items():
                            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                            if not data.empty:
                                portfolio_data[ticker] = data
                                close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
                                initial_prices[ticker]  = float(close.iloc[0])
                                current_prices_[ticker] = float(close.iloc[-1])

                    def get_fx_rate(currency, date, exchange_rates):
                        fx_data = exchange_rates.get(currency) if exchange_rates else None
                        if fx_data is None and currency not in ("CHF",):
                            # Try dynamic fetch for currencies not pre-downloaded
                            pair = get_fx_pair(currency)
                            if pair:
                                try:
                                    fx_data = yf.download(pair, start=start_date, end=end_date, progress=False)
                                    exchange_rates[currency] = fx_data
                                except Exception:
                                    pass
                        return get_fx_rate_for_date(currency, date, fx_data)

                    # ── Risk Metrics ──────────────────────────────────────────
                    st.subheader("📊 Portfolio Risk Metrics")

                    with st.spinner("Calculating Sharpe ratio, Alpha, and Beta…"):
                        all_dates = msci_world.index
                        portfolio_values = pd.Series(index=all_dates, dtype=float)

                        for date in all_dates:
                            daily_value = 0
                            for ticker, info in portfolio_holdings.items():
                                if ticker in portfolio_data and not portfolio_data[ticker].empty:
                                    sd = portfolio_data[ticker]
                                    sc = sd["Close"].iloc[:, 0] if isinstance(sd["Close"], pd.DataFrame) else sd["Close"]
                                    avail = sc.index[sc.index <= date]
                                    price = float(sc.loc[avail[-1]]) if len(avail) > 0 else initial_prices[ticker]
                                    fx    = get_fx_rate(info["currency"], date, exchange_rates)
                                    daily_value += price * info["quantity"] * fx
                            portfolio_values[date] = daily_value

                        port_ret = portfolio_values.pct_change().dropna()
                        msci_close = msci_world["Close"].iloc[:, 0] if isinstance(msci_world["Close"], pd.DataFrame) else msci_world["Close"]
                        msci_ret  = msci_close.pct_change().dropna()

                        common = port_ret.index.intersection(msci_ret.index)
                        pr     = port_ret.loc[common]
                        mr     = msci_ret.loc[common]

                        rf_daily = 0.02 / 252
                        excess   = pr - rf_daily
                        sharpe   = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0

                        from scipy import stats as scipy_stats
                        reg_data = pd.DataFrame({"msci": mr, "portfolio": pr}).dropna()
                        if len(reg_data) > 2:
                            slope, intercept, *_ = scipy_stats.linregress(reg_data["msci"], reg_data["portfolio"])
                            beta        = slope
                            alpha_annual = (1 + intercept) ** 252 - 1
                        else:
                            beta = alpha_annual = 0.0

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sharpe Ratio",         f"{sharpe:.2f}")
                        c2.metric("Beta (vs MSCI World)", f"{beta:.2f}")
                        c3.metric("Alpha (Annualized)",   f"{alpha_annual*100:.2f}%")

                        st.info(f"""
**Interpretation:**
- **Sharpe Ratio ({sharpe:.2f})**: {'Excellent' if sharpe > 2 else 'Good' if sharpe > 1 else 'Moderate' if sharpe > 0 else 'Poor'} risk-adjusted performance
- **Beta ({beta:.2f})**: Portfolio is {'more volatile than' if beta > 1 else 'less volatile than' if beta < 1 else 'as volatile as'} the market
- **Alpha ({alpha_annual*100:.2f}%)**: {'Outperforming' if alpha_annual > 0 else 'Underperforming'} the benchmark by {abs(alpha_annual*100):.2f}% annually
""")

                    st.markdown("---")

                    # =========================================================
                    # CASH-AWARE PORTFOLIO VALUATION — everything in USD
                    # =========================================================
                    # Fund starts with $1,000,000 USD cash.
                    # Each Buy  → cash decreases by (shares × exec_price_in_USD)
                    # Each Sell → cash increases by (shares × exec_price_in_USD)
                    # Daily total NAV = cash_usd + Σ(shares × price × fx_to_usd)
                    # =========================================================

                    FUND_SIZE_USD = 1_000_000.0   # initial capital

                    # ── Helper: get FX rate to USD (not CHF) ─────────────────
                    # We download XXXUSD=X pairs for the valuation period
                    usd_fx_cache = {"USD": None}   # USD→USD = 1.0 always

                    def get_usd_pair(currency):
                        if currency == "USD": return None
                        return f"{currency}USD=X"

                    def fetch_usd_fx(currency):
                        if currency in usd_fx_cache:
                            return usd_fx_cache[currency]
                        pair = get_usd_pair(currency)
                        if pair is None:
                            usd_fx_cache[currency] = None
                            return None
                        try:
                            data = yf.download(pair, start=start_date, end=end_date, progress=False)
                            usd_fx_cache[currency] = data
                        except Exception:
                            usd_fx_cache[currency] = pd.DataFrame()
                        return usd_fx_cache[currency]

                    def get_fx_to_usd(currency, date):
                        """Convert 1 unit of *currency* to USD on *date*."""
                        if currency == "USD":
                            return 1.0
                        fx_data = fetch_usd_fx(currency)
                        if fx_data is None or (hasattr(fx_data, "empty") and fx_data.empty):
                            return FALLBACK_FX_TO_CHF.get(currency, 1.0) / FALLBACK_FX_TO_CHF.get("USD", 0.888)
                        try:
                            close = fx_data["Close"].iloc[:, 0] if isinstance(fx_data["Close"], pd.DataFrame) else fx_data["Close"]
                            avail = close.index[close.index <= date]
                            return float(close.loc[avail[-1]]) if len(avail) > 0 else float(close.iloc[0])
                        except Exception:
                            return 1.0

                    # Pre-fetch USD FX for all portfolio currencies
                    with st.spinner("Fetching USD FX rates…"):
                        for info in portfolio_holdings.values():
                            fetch_usd_fx(info["currency"])

                    # ── Reconstruct cash balance day-by-day from transactions ─
                    # Sort all transactions chronologically
                    all_tx = _tx.sort_values("Date").reset_index(drop=True)

                    def get_stock_price_on(ticker, date, stock_data_dict):
                        """Get closing price of ticker on or before date."""
                        if ticker not in stock_data_dict:
                            return None
                        d = stock_data_dict[ticker]
                        cl = d["Close"].iloc[:,0] if isinstance(d["Close"], pd.DataFrame) else d["Close"]
                        avail = cl.index[cl.index <= date]
                        return float(cl.loc[avail[-1]]) if len(avail) > 0 else None

                    dates = msci_close.index

                    # Build daily NAV series
                    nav_usd          = []
                    equity_usd_daily = []
                    cash_usd_daily   = []

                    # Running cash starts at fund size; depleted/replenished by trades
                    cash_usd = FUND_SIZE_USD

                    # Running share count per ticker (adjusted as trades happen)
                    live_shares = {ticker: 0 for ticker in portfolio_holdings}

                    # Also track tickers that appear in transactions but may be fully sold
                    for ticker in all_tx["Ticker"].unique():
                        if ticker not in live_shares:
                            live_shares[ticker] = 0

                    # Index transactions by date for fast lookup
                    tx_by_date = {}
                    for _, tx_row in all_tx.iterrows():
                        d = tx_row["Date"].normalize()
                        tx_by_date.setdefault(d, []).append(tx_row)

                    for date in dates:
                        date_norm = pd.Timestamp(date).normalize()

                        # Process any transactions that happened ON this date
                        for tx_row in tx_by_date.get(date_norm, []):
                            ticker  = tx_row["Ticker"]
                            qty     = int(tx_row["Quantity"])
                            action  = tx_row["Action"]
                            # Use closing price on trade date converted to USD
                            price_native = get_stock_price_on(ticker, date, portfolio_data)
                            if price_native is None:
                                price_native = portfolio_holdings.get(ticker, {}).get("purchase_price", 0)
                            ccy      = portfolio_holdings.get(ticker, {}).get("currency", "USD")
                            price_usd = price_native * get_fx_to_usd(ccy, date)

                            if action == "Buy":
                                cash_usd          -= price_usd * qty
                                live_shares[ticker] = live_shares.get(ticker, 0) + qty
                            elif action == "Sell":
                                cash_usd          += price_usd * qty
                                live_shares[ticker] = max(0, live_shares.get(ticker, 0) - qty)

                        # Value equity positions in USD
                        equity_usd = 0.0
                        for ticker, shares in live_shares.items():
                            if shares <= 0:
                                continue
                            price_native = get_stock_price_on(ticker, date, portfolio_data)
                            if price_native is None:
                                continue
                            ccy       = portfolio_holdings.get(ticker, {}).get("currency", "USD")
                            price_usd = price_native * get_fx_to_usd(ccy, date)
                            equity_usd += price_usd * shares

                        total_nav = cash_usd + equity_usd
                        nav_usd.append(total_nav)
                        equity_usd_daily.append(equity_usd)
                        cash_usd_daily.append(cash_usd)

                    nav_series    = pd.Series(nav_usd,          index=dates)
                    equity_series = pd.Series(equity_usd_daily, index=dates)
                    cash_series   = pd.Series(cash_usd_daily,   index=dates)

                    # ── Performance chart ─────────────────────────────────────
                    st.subheader("📈 Portfolio vs MSCI World Performance")

                    port_norm = (nav_series / FUND_SIZE_USD * 100)
                    msci_norm = (msci_close / msci_close.iloc[0] * 100)

                    fig_perf = go.Figure()

                    # Total NAV
                    fig_perf.add_trace(go.Scatter(
                        x=dates, y=port_norm, mode="lines",
                        name="Total NAV (equity + cash)",
                        line=dict(color="#0F1D64", width=3),
                        hovertemplate="<b>Total NAV</b><br>%{x}<br>$%{customdata:,.0f}<extra></extra>",
                        customdata=nav_series.values,
                    ))
                    # MSCI World
                    fig_perf.add_trace(go.Scatter(
                        x=dates, y=msci_norm, mode="lines",
                        name="MSCI World (URTH)",
                        line=dict(color="#FF6B6B", width=2),
                        hovertemplate="<b>MSCI World</b><br>%{x}<br>Index: %{y:.2f}<extra></extra>",
                    ))

                    fig_perf.update_layout(
                        title="Fund NAV vs MSCI World (Base 100 = $1,000,000 USD)",
                        xaxis_title="Date", yaxis_title="Index Value (Base 100)",
                        hovermode="x unified", height=520, template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)

                    # ── Key metrics ───────────────────────────────────────────
                    final_nav    = nav_series.iloc[-1]
                    final_cash   = cash_series.iloc[-1]
                    final_equity = equity_series.iloc[-1]
                    port_ret_total = (final_nav - FUND_SIZE_USD) / FUND_SIZE_USD * 100
                    msci_ret_total = (msci_close.iloc[-1] - msci_close.iloc[0]) / msci_close.iloc[0] * 100
                    outperf        = port_ret_total - msci_ret_total
                    cash_pct       = final_cash / final_nav * 100

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Fund NAV",          f"${final_nav:,.0f}",
                              delta=f"{port_ret_total:.2f}% vs $1M")
                    c2.metric("Equity Value",       f"${final_equity:,.0f}",
                              delta=f"{final_equity/final_nav*100:.1f}% of NAV")
                    c3.metric("Cash (USD)",         f"${final_cash:,.0f}",
                              delta=f"{cash_pct:.1f}% of NAV")
                    c4.metric("vs MSCI World",      f"{outperf:+.2f}%",
                              delta=f"{outperf:+.2f}%")

                    st.info(f"""
**Fund Summary:** Initial capital **$1,000,000 USD** | 
Fund return **{port_ret_total:+.2f}%** | 
MSCI World **{msci_ret_total:+.2f}%** | 
Alpha **{outperf:+.2f}%** | 
Cash drag **{cash_pct:.1f}%** of NAV
""")

                    # ── Treemap (USD, includes cash) ───────────────────────────
                    st.subheader("🗺️ Portfolio Composition Treemap")
                    treemap_data = []
                    final_date   = dates[-1]

                    for ticker, shares in live_shares.items():
                        if shares <= 0:
                            continue
                        info = portfolio_holdings.get(ticker)
                        if info is None:
                            continue
                        price_native = get_stock_price_on(ticker, final_date, portfolio_data)
                        if price_native is None:
                            continue
                        ccy       = info["currency"]
                        price_usd = price_native * get_fx_to_usd(ccy, final_date)
                        curr_val  = price_usd * shares

                        # Cost basis in USD
                        buy_tx = all_tx[(all_tx["Ticker"] == ticker) & (all_tx["Action"] == "Buy")]
                        cost_usd = 0.0
                        for _, brow in buy_tx.iterrows():
                            bp = get_stock_price_on(ticker, brow["Date"], portfolio_data)
                            if bp is None: bp = info["purchase_price"]
                            cost_usd += bp * get_fx_to_usd(ccy, brow["Date"]) * int(brow["Quantity"])
                        avg_cost = cost_usd / shares if shares > 0 else 0
                        perf = (price_usd - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0

                        treemap_data.append({
                            "ticker": ticker, "name": info.get("name", ticker),
                            "weight": curr_val / final_nav * 100,
                            "performance": perf, "value": curr_val,
                            "currency": ccy,
                        })

                    # Add cash as a treemap entry
                    treemap_data.append({
                        "ticker": "CASH", "name": "Cash (USD)",
                        "weight": final_cash / final_nav * 100,
                        "performance": 0.0, "value": final_cash,
                        "currency": "USD",
                    })

                    treemap_df = pd.DataFrame(treemap_data)
                    fig_tm = px.treemap(
                        treemap_df, path=[px.Constant("Portfolio"), "name"],
                        values="weight", color="performance",
                        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        hover_data={"weight":":.2f","performance":":.2f","value":":,.0f","currency":True},
                        labels={"weight":"Weight (%)","performance":"Return (%)","value":"Value (USD)","currency":"Currency"},
                    )
                    fig_tm.update_traces(
                        textposition="middle center",
                        texttemplate="<b>%{label}</b><br>%{value:.1f}%",
                        hovertemplate="<b>%{label}</b><br>Weight: %{customdata[0]:.2f}%<br>Return: %{customdata[1]:.2f}%<br>Value: $%{customdata[2]:,.0f}<br>Ccy: %{customdata[3]}<extra></extra>",
                    )
                    fig_tm.update_layout(height=600, margin=dict(t=50,l=0,r=0,b=0))
                    st.plotly_chart(fig_tm, use_container_width=True)

                    # ── Holdings table (USD, includes cash row) ───────────────
                    st.subheader("📋 Holdings Details (USD)")
                    ht = treemap_df.copy()
                    ht["Return (%)"]   = ht["performance"].apply(lambda x: f"{x:.2f}%")
                    ht["Weight (%)"]   = ht["weight"].apply(lambda x: f"{x:.2f}%")
                    ht["Value (USD)"]  = ht["value"].apply(lambda x: f"${x:,.0f}")
                    ht = ht[["name","ticker","currency","Weight (%)","Return (%)","Value (USD)"]]
                    ht.columns = ["Company","Ticker","Currency","Weight","Return","Value (USD)"]
                    st.dataframe(ht.sort_values("Return", ascending=False),
                                 use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    import traceback; st.code(traceback.format_exc())

    # -------------------------------------------------------------------------
    # HOME - Portfolio Structure Analysis
    # -------------------------------------------------------------------------
    elif home_tab == "Portfolio Structure Analysis":
        st.header("🏗️ Portfolio Structure Analysis")

        # ── Country/region auto-fetched from yfinance HQ data ──────────────
        @st.cache_data(show_spinner=False)
        def fetch_hq_countries(tickers: tuple) -> dict:
            """
            For each ticker fetch headquarters country from yfinance .info.
            Returns {ticker: country_name}.
            Falls back to exchange-based inference if yfinance field is missing.
            """
            EXCHANGE_COUNTRY_FALLBACK = {
                ".NS": "India", ".BO": "India",
                ".DE": "Germany", ".F": "Germany",
                ".SW": "Switzerland",
                ".PA": "France",
                ".AS": "Netherlands",
                ".L":  "United Kingdom",
                ".HK": "Hong Kong",
                ".AX": "Australia",
                ".TO": "Canada", ".V": "Canada",
                ".T":  "Japan",
                ".KS": "South Korea",
                ".SS": "China", ".SZ": "China",
                ".BR": "Belgium",
                ".ST": "Sweden",
                ".CO": "Denmark",
                ".OL": "Norway",
                ".HE": "Finland",
                ".LS": "Portugal",
                ".MC": "Spain",
                ".MI": "Italy",
                ".LI": "Liechtenstein",
            }
            result = {}
            for ticker in tickers:
                try:
                    info = yf.Ticker(ticker).info
                    country = info.get("country") or info.get("headquartersCountry")
                    if country:
                        result[ticker] = country
                        continue
                except Exception:
                    pass
                # Fallback: infer from ticker suffix
                for suffix, country in EXCHANGE_COUNTRY_FALLBACK.items():
                    if ticker.upper().endswith(suffix.upper()):
                        result[ticker] = country
                        break
                else:
                    result[ticker] = "United States"  # bare tickers (no suffix) are typically US
            return result

        # Comprehensive region and ISO mappings
        region_mapping = {
            "India": "APAC", "China": "APAC", "Hong Kong": "APAC",
            "Japan": "APAC", "South Korea": "APAC", "Australia": "APAC",
            "Singapore": "APAC", "Taiwan": "APAC", "New Zealand": "APAC",
            "United States": "Americas", "Canada": "Americas",
            "Brazil": "Americas", "Mexico": "Americas", "Argentina": "Americas",
            "Germany": "EMEA", "France": "EMEA", "Switzerland": "EMEA",
            "Netherlands": "EMEA", "United Kingdom": "EMEA", "Sweden": "EMEA",
            "Denmark": "EMEA", "Norway": "EMEA", "Finland": "EMEA",
            "Belgium": "EMEA", "Spain": "EMEA", "Italy": "EMEA",
            "Portugal": "EMEA", "Ireland": "EMEA", "Luxembourg": "EMEA",
            "Liechtenstein": "EMEA", "Austria": "EMEA", "Israel": "EMEA",
            "South Africa": "EMEA", "Saudi Arabia": "EMEA", "UAE": "EMEA",
        }
        country_iso = {
            "India": "IND", "China": "CHN", "Hong Kong": "HKG",
            "Japan": "JPN", "South Korea": "KOR", "Australia": "AUS",
            "Singapore": "SGP", "Taiwan": "TWN", "New Zealand": "NZL",
            "United States": "USA", "Canada": "CAN",
            "Brazil": "BRA", "Mexico": "MEX", "Argentina": "ARG",
            "Germany": "DEU", "France": "FRA", "Switzerland": "CHE",
            "Netherlands": "NLD", "United Kingdom": "GBR", "Sweden": "SWE",
            "Denmark": "DNK", "Norway": "NOR", "Finland": "FIN",
            "Belgium": "BEL", "Spain": "ESP", "Italy": "ITA",
            "Portugal": "PRT", "Ireland": "IRL", "Luxembourg": "LUX",
            "Liechtenstein": "LIE", "Austria": "AUT", "Israel": "ISR",
            "South Africa": "ZAF", "Saudi Arabia": "SAU", "UAE": "ARE",
        }

        with st.spinner("Fetching HQ countries from yfinance…"):
            country_mapping = fetch_hq_countries(tuple(sorted(portfolio_holdings.keys())))

        try:
            with st.spinner("Loading portfolio structure…"):
                ex_rates_cur = {}
                used_currencies_struct = {info["currency"] for info in portfolio_holdings.values()}
                for currency in used_currencies_struct:
                    if currency == "CHF":
                        ex_rates_cur[currency] = 1.0
                    else:
                        pair = get_fx_pair(currency)
                        if pair:
                            try:
                                fx_d = yf.Ticker(pair).history(period="1d")
                                ex_rates_cur[currency] = float(fx_d["Close"].iloc[-1]) if not fx_d.empty else FALLBACK_FX_TO_CHF.get(currency, 1.0)
                            except Exception:
                                ex_rates_cur[currency] = FALLBACK_FX_TO_CHF.get(currency, 1.0)
                        else:
                            ex_rates_cur[currency] = 1.0

                cur_prices = {}
                co_info    = {}
                for ticker, info in portfolio_holdings.items():
                    try:
                        tk   = yf.Ticker(ticker)
                        hist = tk.history(period="1d")
                        cur_prices[ticker] = float(hist["Close"].iloc[-1]) if not hist.empty else info["purchase_price"]
                        try:
                            tki = tk.info
                            co_info[ticker] = {"market_cap": tki.get("marketCap", 0)}
                        except Exception:
                            co_info[ticker] = {"market_cap": 0}
                    except Exception:
                        cur_prices[ticker] = info["purchase_price"]
                        co_info[ticker]    = {"market_cap": 0}

            def classify_mc(mc):
                if mc == 0:            return "Unknown"
                elif mc >= 10e9:       return "Large"
                elif mc >= 2e9:        return "Mid"
                else:                  return "Small"

            holdings_analysis = []
            total_chf = 0
            for ticker, info in portfolio_holdings.items():
                if ticker in cur_prices:
                    fx  = ex_rates_cur[info["currency"]]
                    val = cur_prices[ticker] * info["quantity"] * fx
                    total_chf += val
                    country = country_mapping.get(ticker) or "United States"
                    holdings_analysis.append({
                        "ticker": ticker, "name": info["name"], "sector": info["sector"],
                        "country": country, "country_iso": country_iso.get(country, ""),
                        "region": region_mapping.get(country, "Unknown"),
                        "market_cap": classify_mc(co_info.get(ticker, {}).get("market_cap", 0)),
                        "currency": info["currency"], "value_chf": val, "weight": 0,
                    })

            for h in holdings_analysis:
                h["weight"] = h["value_chf"] / total_chf * 100

            df_an = pd.DataFrame(holdings_analysis)
            blue  = ["#0F1D64","#1E3A8A","#3B82F6","#60A5FA","#93C5FD","#DBEAFE"]

            # Sector donut
            st.subheader("📊 Sector Distribution")
            sec_alloc = df_an.groupby("sector")["weight"].sum().reset_index()
            sec_alloc.columns = ["Sector","Weight (%)"]
            c1, c2 = st.columns([2,1])
            with c1:
                fig_s = px.pie(sec_alloc, values="Weight (%)", names="Sector",
                               title="Portfolio Allocation by Sector",
                               color_discrete_sequence=blue, hole=0.4)
                fig_s.update_traces(textposition="inside", textinfo="percent+label")
                fig_s.update_layout(height=400)
                st.plotly_chart(fig_s, use_container_width=True)
            with c2:
                st.markdown("### Sector Breakdown")
                for _, r in sec_alloc.sort_values("Weight (%)", ascending=False).iterrows():
                    st.metric(r["Sector"], f"{r['Weight (%)']:.1f}%")

            # Geo map
            st.subheader("🌍 Geographical Distribution")
            c1, c2 = st.columns([2,1])
            with c1:
                cnt_alloc = df_an.groupby(["country","country_iso"])["weight"].sum().reset_index()
                cnt_alloc.columns = ["Country","ISO","Weight (%)"]
                fig_map = px.choropleth(cnt_alloc, locations="ISO", color="Weight (%)",
                                        hover_name="Country",
                                        hover_data={"ISO":False,"Weight (%)":":.2f"},
                                        color_continuous_scale=[[0,"#DBEAFE"],[0.25,"#93C5FD"],
                                                                 [0.5,"#60A5FA"],[0.75,"#3B82F6"],[1,"#0F1D64"]],
                                        title="Geographic Distribution")
                fig_map.update_geos(showcountries=True, countrycolor="lightgray",
                                    showcoastlines=True, projection_type="natural earth")
                fig_map.update_layout(height=450, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_map, use_container_width=True)
            with c2:
                st.markdown("### Regional Allocation")
                reg_alloc = df_an.groupby("region")["weight"].sum().reset_index()
                reg_alloc.columns = ["Region","Weight (%)"]
                for _, r in reg_alloc.sort_values("Weight (%)", ascending=False).iterrows():
                    st.metric(r["Region"], f"{r['Weight (%)']:.1f}%")
                st.markdown("---")
                st.markdown("### Top Countries")
                st.dataframe(cnt_alloc[["Country","Weight (%)"]].sort_values("Weight (%)", ascending=False)
                             .style.format({"Weight (%)":"{:.1f}%"}), use_container_width=True, hide_index=True)

            # Market cap & currency
            st.subheader("📈 Additional Analysis")
            c1, c2 = st.columns(2)
            with c1:
                mc_alloc = df_an.groupby("market_cap")["weight"].sum().reset_index()
                mc_alloc.columns = ["Market Cap","Weight (%)"]
                fig_mc = px.bar(mc_alloc, x="Market Cap", y="Weight (%)",
                                title="Market Cap Distribution", color="Weight (%)",
                                color_continuous_scale=[[0,"#DBEAFE"],[0.5,"#3B82F6"],[1,"#0F1D64"]])
                fig_mc.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)
            with c2:
                cur_alloc = df_an.groupby("currency")["weight"].sum().reset_index()
                cur_alloc.columns = ["Currency","Weight (%)"]
                fig_cur = px.pie(cur_alloc, values="Weight (%)", names="Currency",
                                 title="Currency Exposure", color_discrete_sequence=blue)
                fig_cur.update_traces(textposition="inside", textinfo="percent+label")
                fig_cur.update_layout(height=350)
                st.plotly_chart(fig_cur, use_container_width=True)

            # Concentration
            st.subheader("🎯 Concentration Metrics")
            df_sorted   = df_an.sort_values("weight", ascending=False)
            top5_conc   = df_sorted.head(5)["weight"].sum()
            top10_conc  = df_sorted.head(10)["weight"].sum()
            hhi         = (df_an["weight"] ** 2).sum()
            c1, c2, c3  = st.columns(3)
            c1.metric("Top 5 Holdings",  f"{top5_conc:.1f}%")
            c2.metric("Top 10 Holdings", f"{top10_conc:.1f}%")
            with c3:
                st.metric("HHI Index", f"{hhi:.0f}")
                st.caption("✅ Well diversified" if hhi < 1000 else "⚠️ Moderately concentrated" if hhi < 1800 else "🔴 Highly concentrated")

            st.markdown("### Top 10 Holdings by Weight")
            th = df_sorted[["name","sector","country","weight"]].head(10).copy()
            th.columns = ["Company","Sector","Country","Weight (%)"]
            th["Weight (%)"] = th["Weight (%)"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(th, use_container_width=True, hide_index=True)

            st.subheader("📋 Portfolio Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Holdings",  len(df_an))
            c2.metric("Total Value",     f"CHF {total_chf:,.2f}")
            c3.metric("Sectors Covered", df_an["sector"].nunique())
            c4.metric("Countries",       df_an["country"].nunique())

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback; st.code(traceback.format_exc())

    # -------------------------------------------------------------------------
    # HOME - Forecast
    # -------------------------------------------------------------------------
    elif home_tab == "Forecast":
        st.header("🔮 Portfolio Forecast")

        with st.spinner("Loading stock data…"):
            stock_data = {}
            for ticker in portfolio_holdings:
                try:
                    hist = yf.Ticker(ticker).history(period="1y")
                    if not hist.empty:
                        stock_data[ticker] = {"current_price": hist["Close"].iloc[-1], "historical": hist}
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")

        forecast_method = st.tabs(["Monte Carlo Simulation", "Analyst Consensus", "DCF Analysis"])

        # ── Monte Carlo ──────────────────────────────────────────────────────
        with forecast_method[0]:
            st.subheader("📊 Monte Carlo Simulation")
            st.markdown("Probabilistic portfolio performance projections based on historical volatility")

            c1, c2, c3 = st.columns(3)
            with c1: num_sim = st.number_input("Simulations",   100, 10000, 1000, 100)
            with c2: time_h  = st.number_input("Horizon (days)", 30, 1825,   252,  30)
            with c3:
                tot_val  = sum(stock_data[t]["current_price"] * portfolio_holdings[t]["quantity"]
                               for t in portfolio_holdings if t in stock_data)
                init_inv = st.number_input("Initial Portfolio Value ($)", 1000, value=int(tot_val), step=1000)

            if st.button("Run Monte Carlo Simulation", type="primary"):
                with st.spinner("Running simulations…"):
                    pv = {t: stock_data[t]["current_price"] * portfolio_holdings[t]["quantity"]
                          for t in portfolio_holdings if t in stock_data}
                    tot = sum(pv.values())
                    wts = {t: pv[t] / tot for t in pv}
                    ret_data = {t: stock_data[t]["historical"]["Close"].pct_change().dropna()
                                for t in portfolio_holdings if t in stock_data}
                    ret_df = pd.DataFrame(ret_data).dropna()

                    if ret_df.empty or len(ret_df) < 30:
                        st.error("Not enough data.")
                    else:
                        mu  = ret_df.mean()
                        cov = ret_df.cov() + np.eye(len(ret_df.columns)) * 1e-8
                        np.random.seed(42)
                        sims = np.zeros((time_h, num_sim))
                        tl   = list(ret_df.columns)

                        for i in range(num_sim):
                            vals = [init_inv]
                            for _ in range(time_h):
                                try:    rr = mu.values + np.linalg.cholesky(cov) @ np.random.standard_normal(len(tl))
                                except: rr = np.random.normal(mu.values, np.sqrt(np.diag(cov)))
                                pr = sum(wts.get(tl[j], 0) * rr[j] for j in range(len(tl)))
                                vals.append(vals[-1] * (1 + pr))
                            sims[:, i] = vals[1:]

                        p5, p50, p95 = (np.percentile(sims, p, axis=1) for p in [5, 50, 95])

                        fig = go.Figure()
                        for i in range(0, num_sim, max(1, num_sim // 100)):
                            fig.add_trace(go.Scatter(x=list(range(time_h)), y=sims[:, i],
                                                     mode="lines", line=dict(width=0.5, color="lightgray"),
                                                     showlegend=False, hoverinfo="skip"))
                        for arr, name, color in [(p5,"5th %ile (Bear)","red"),
                                                 (p50,"50th %ile (Base)","blue"),
                                                 (p95,"95th %ile (Bull)","green")]:
                            fig.add_trace(go.Scatter(x=list(range(time_h)), y=arr,
                                                     mode="lines", name=name, line=dict(color=color, width=3)))
                        fig.update_layout(title=f"Monte Carlo: {num_sim} scenarios over {time_h} days",
                                          xaxis_title="Days", yaxis_title="Portfolio Value ($)",
                                          height=500, hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

                        c1, c2, c3, c4 = st.columns(4)
                        for col, val, label in [(c1,p5[-1],"Bear (5th)"),(c2,p50[-1],"Base (50th)"),
                                                (c3,p95[-1],"Bull (95th)"),(c4,np.mean(sims[-1,:]),"Expected (Mean)")]:
                            col.metric(label, f"${val:,.0f}", f"{(val-init_inv)/init_inv*100:.1f}%")

        # ── Analyst Consensus ────────────────────────────────────────────────
        with forecast_method[1]:
            st.subheader("📈 Analyst Consensus Forecast")
            analyst_data = []
            for ticker in portfolio_holdings:
                if ticker not in stock_data: continue
                try:
                    info_ = yf.Ticker(ticker).info
                    cp    = stock_data[ticker]["current_price"]
                    tp    = info_.get("targetMeanPrice")
                    if tp:
                        up  = (tp - cp) / cp * 100
                        cv  = cp * portfolio_holdings[ticker]["quantity"]
                        pv_ = cv * (1 + up / 100)
                        analyst_data.append({"Ticker":ticker,"Current Price":cp,"Target Price":tp,
                                             "Upside/Downside":up,"Analysts":info_.get("numberOfAnalystOpinions",0),
                                             "Recommendation":info_.get("recommendationKey","N/A").upper(),
                                             "Current Value":cv,"Projected Value":pv_,"Potential Gain":pv_-cv})
                except Exception as e:
                    st.warning(f"Could not fetch analyst data for {ticker}: {e}")

            if analyst_data:
                da = pd.DataFrame(analyst_data)
                tc = da["Current Value"].sum(); tp_ = da["Projected Value"].sum()
                tg = da["Potential Gain"].sum(); wu  = tg / tc * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Portfolio Value",   f"${tc:,.0f}")
                c2.metric("Projected Value (12M)",     f"${tp_:,.0f}", f"{wu:.1f}%")
                c3.metric("Potential Gain",            f"${tg:,.0f}")
                dd = da.copy()
                for col in ["Current Price","Target Price"]:         dd[col] = dd[col].apply(lambda x: f"${x:.2f}")
                dd["Upside/Downside"]  = dd["Upside/Downside"].apply(lambda x: f"{x:.1f}%")
                for col in ["Current Value","Projected Value","Potential Gain"]: dd[col] = dd[col].apply(lambda x: f"${x:,.0f}")
                st.dataframe(dd, use_container_width=True, hide_index=True)
                fig = go.Figure(go.Bar(x=da["Ticker"], y=da["Upside/Downside"],
                                       marker_color=["green" if x > 0 else "red" for x in da["Upside/Downside"]],
                                       text=[f"{x:.1f}%" for x in da["Upside/Downside"]], textposition="outside"))
                fig.update_layout(title="Analyst Consensus: Upside/Downside", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No analyst consensus data available.")

        # ── DCF ──────────────────────────────────────────────────────────────
        with forecast_method[2]:
            st.subheader("💰 DCF Analysis")
            st.info("Simplified DCF model. Consult financial statements for accurate valuations.")
            sel = st.selectbox("Select Stock for DCF Analysis",
                               [t for t in portfolio_holdings if t in stock_data])
            if sel:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Input Parameters")
                    try:
                        cf = yf.Ticker(sel).cashflow
                        fcf_default = int(abs(cf.loc["Free Cash Flow"].iloc[0])) if not cf.empty and "Free Cash Flow" in cf.index else 1_000_000_000
                    except Exception:
                        fcf_default = 1_000_000_000
                    fcf_in = st.number_input("Current Free Cash Flow ($)", 0, value=fcf_default, step=1_000_000)
                    gr     = st.slider("Growth Rate (%)",       -10.0, 50.0, 5.0, 0.5)
                    py     = st.number_input("Projection (years)", 1, 10, 5)
                    wacc_  = st.slider("WACC (%)",               1.0, 20.0, 10.0, 0.5)
                    tg     = st.slider("Terminal Growth (%)",    0.0,  5.0,  2.5, 0.1)
                    try:    so = int(yf.Ticker(sel).info.get("sharesOutstanding", 1_000_000_000))
                    except: so = 1_000_000_000
                    shares = st.number_input("Shares Outstanding", 1_000_000, value=so, step=1_000_000)

                with c2:
                    st.subheader("DCF Calculation")
                    pf = [fcf_in * (1 + gr / 100) ** y for y in range(1, py + 1)]
                    pv = [pf[i] / (1 + wacc_ / 100) ** (i + 1) for i in range(py)]
                    tv = pf[-1] * (1 + tg / 100) / (wacc_ / 100 - tg / 100)
                    pv_tv = tv / (1 + wacc_ / 100) ** py
                    ev   = sum(pv) + pv_tv
                    fv   = ev / shares
                    cp   = stock_data[sel]["current_price"]
                    ud   = (fv - cp) / cp * 100

                    m1, m2 = st.columns(2)
                    m1.metric("Fair Value per Share", f"${fv:.2f}")
                    m1.metric("Enterprise Value",     f"${ev/1e9:.2f}B")
                    m2.metric("Current Price",        f"${cp:.2f}")
                    m2.metric("Upside/Downside",      f"{ud:.1f}%", delta=f"{ud:.1f}%")

                    dcf_rows = [{"Year": y+1, "Projected FCF": f"${pf[y]/1e6:.1f}M",
                                 "Discount Factor": f"{1/(1+wacc_/100)**(y+1):.4f}",
                                 "Present Value": f"${pv[y]/1e6:.1f}M"} for y in range(py)]
                    st.dataframe(pd.DataFrame(dcf_rows), use_container_width=True, hide_index=True)

                    if ud > 20:      st.success(f"💡 **Undervalued** by {ud:.1f}%")
                    elif ud < -20:   st.error(f"💡 **Overvalued** by {abs(ud):.1f}%")
                    else:            st.info("💡 **Fairly valued** (within ±20%)")

# =============================================================================
# SECTOR PAGES
# =============================================================================
elif main_page in ["TMT Sector","FIG Sector","Industrials Sector",
                   "PUI Sector","Consumer Goods Sector","Healthcare Sector"]:

    sector_name = main_page.replace(" Sector", "")
    st.title(f"🏢 {sector_name} Sector Analysis")

    sector_holdings = {k: v for k, v in portfolio_holdings.items() if v["sector"] == sector_name}

    if not sector_holdings:
        st.warning(f"No holdings found in {sector_name} sector.")
    else:
        if "sector_tab" not in st.session_state:
            st.session_state.sector_tab = "Performance Analysis"

        st.markdown("### Select Analysis Type")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📈 Performance Analysis", key=f"perf_{sector_name}", use_container_width=True):
                st.session_state.sector_tab = "Performance Analysis"; st.rerun()
        with c2:
            if st.button("💰 Financial Analysis",   key=f"fin_{sector_name}",  use_container_width=True):
                st.session_state.sector_tab = "Financial Analysis";  st.rerun()
        with c3:
            if st.button("🏢 Company Specific",     key=f"spec_{sector_name}", use_container_width=True):
                st.session_state.sector_tab = "Company Specific";    st.rerun()

        st.markdown("---")
        sector_tab = st.session_state.sector_tab

        # ── Performance Analysis ─────────────────────────────────────────────
        if sector_tab == "Performance Analysis":
            st.header(f"📈 {sector_name} - Performance Analysis")

            benchmarks = {"TMT":"XLK","FIG":"XLF","Industrials":"XLI",
                          "PUI":"XLB","Consumer Goods":"XLP","Healthcare":"XLV"}
            bm_ticker  = benchmarks.get(sector_name, "URTH")

            c1, c2 = st.columns(2)
            with c1: start_date = st.date_input("Start Date", pd.to_datetime("2025-11-06"), key=f"s_{sector_name}")
            with c2: end_date   = st.date_input("End Date",   pd.to_datetime("today"),       key=f"e_{sector_name}")

            if st.button("📊 Generate Performance Analysis", type="primary", key=f"gen_{sector_name}"):
                if start_date >= end_date:
                    st.error("Start date must be before end date.")
                else:
                    try:
                        from scipy import stats as scipy_stats
                        with st.spinner(f"Fetching {sector_name} data…"):
                            bm_data = yf.download(bm_ticker, start=start_date, end=end_date, progress=False)
                            used_currencies_s = {info["currency"] for info in sector_holdings.values()}
                            ex_rates = {}
                            for currency in used_currencies_s:
                                if currency == "CHF":
                                    ex_rates[currency] = None
                                else:
                                    pair = get_fx_pair(currency)
                                    if pair:
                                        try:
                                            ex_rates[currency] = yf.download(pair, start=start_date, end=end_date, progress=False)
                                        except Exception:
                                            ex_rates[currency] = pd.DataFrame()
                                    else:
                                        ex_rates[currency] = None
                            sector_data  = {}
                            init_prices  = {}
                            cur_prices_s = {}
                            for ticker, info in sector_holdings.items():
                                d = yf.download(ticker, start=start_date, end=end_date, progress=False)
                                if not d.empty:
                                    sector_data[ticker] = d
                                    cl = d["Close"].iloc[:,0] if isinstance(d["Close"],pd.DataFrame) else d["Close"]
                                    init_prices[ticker]  = float(cl.iloc[0])
                                    cur_prices_s[ticker] = float(cl.iloc[-1])

                        def gfx(currency, date, er):
                            fx_d = er.get(currency) if er else None
                            if fx_d is None and currency != "CHF":
                                pair = get_fx_pair(currency)
                                if pair:
                                    try:
                                        fx_d = yf.download(pair, start=start_date, end=end_date, progress=False)
                                        er[currency] = fx_d
                                    except Exception:
                                        pass
                            return get_fx_rate_for_date(currency, date, fx_d)

                        all_dates   = bm_data.index
                        sec_vals    = pd.Series(index=all_dates, dtype=float)
                        for date in all_dates:
                            dv = 0
                            for ticker, info in sector_holdings.items():
                                if ticker in sector_data:
                                    sd = sector_data[ticker]
                                    sc = sd["Close"].iloc[:,0] if isinstance(sd["Close"],pd.DataFrame) else sd["Close"]
                                    av = sc.index[sc.index <= date]
                                    pr = float(sc.loc[av[-1]]) if len(av) > 0 else init_prices[ticker]
                                    dv += pr * info["quantity"] * gfx(info["currency"], date, ex_rates)
                            sec_vals[date] = dv

                        sec_ret = sec_vals.pct_change().dropna()
                        bm_close = bm_data["Close"].iloc[:,0] if isinstance(bm_data["Close"],pd.DataFrame) else bm_data["Close"]
                        bm_ret  = bm_close.pct_change().dropna()
                        common  = sec_ret.index.intersection(bm_ret.index)
                        sr      = sec_ret.loc[common]; br = bm_ret.loc[common]

                        trs = ((sec_vals.iloc[-1]/sec_vals.iloc[0])-1)*100
                        trb = ((bm_close.iloc[-1]/bm_close.iloc[0])-1)*100

                        rf    = 0.02/252
                        exc   = sr - rf
                        sharpe = (exc.mean()/exc.std())*np.sqrt(252) if exc.std() != 0 else 0

                        rd = pd.DataFrame({"bm":br,"sec":sr}).dropna()
                        if len(rd) > 2:
                            sl, ic, *_ = scipy_stats.linregress(rd["bm"], rd["sec"])
                            beta = sl; alpha_a = ic * 252
                        else:
                            beta = alpha_a = 0.0

                        vol_s = sr.std() * np.sqrt(252) * 100
                        vol_b = br.std() * np.sqrt(252) * 100
                        cum   = (1+sr).cumprod(); mdd = ((cum - cum.expanding().max())/cum.expanding().max()).min()*100
                        act   = sr - br; te = act.std()*np.sqrt(252)
                        ir    = (act.mean()*252)/te if te != 0 else 0

                        st.subheader("🎯 Key Performance Metrics")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Return (Sector)",    f"{trs:.2f}%", delta=f"{trs-trb:.2f}% vs benchmark")
                        c1.metric("Sharpe Ratio",             f"{sharpe:.2f}")
                        c2.metric("Total Return (Benchmark)", f"{trb:.2f}%")
                        c2.metric("Beta",                     f"{beta:.2f}")
                        c3.metric("Volatility (Sector)",      f"{vol_s:.2f}%")
                        c3.metric("Max Drawdown",             f"{mdd:.2f}%")
                        c4.metric("Volatility (Benchmark)",   f"{vol_b:.2f}%")
                        c4.metric("Information Ratio",        f"{ir:.2f}")

                        c1, c2 = st.columns(2)
                        c1.metric("Alpha (Annualized)",  f"{alpha_a*100:.2f}%")
                        c2.metric("Tracking Error",      f"{te*100:.2f}%")

                        st.subheader(f"📊 {sector_name} vs {bm_ticker} Performance")
                        sn = (sec_vals/sec_vals.iloc[0])*100
                        bn = (bm_close/bm_close.iloc[0])*100
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=sn.index, y=sn, name=f"{sector_name} Portfolio", mode="lines"))
                        fig.add_trace(go.Scatter(x=bn.index, y=bn, name=f"{bm_ticker} Benchmark",   mode="lines"))
                        fig.update_xaxes(rangeselector=dict(buttons=[
                            dict(count=1,label="1M",step="month",stepmode="backward"),
                            dict(count=6,label="6M",step="month",stepmode="backward"),
                            dict(count=1,label="1Y",step="year",stepmode="backward"),
                            dict(step="all",label="All"),
                        ]))
                        fig.update_layout(yaxis_title="Normalized Value (Base=100)", xaxis_title="Date")
                        st.plotly_chart(fig, use_container_width=True)

                        # Holdings breakdown
                        st.subheader("🏆 Holdings Performance Breakdown")
                        hp = []
                        for ticker, info in sector_holdings.items():
                            if ticker in sector_data:
                                ret_    = ((cur_prices_s[ticker]/init_prices[ticker])-1)*100
                                iv_chf  = init_prices[ticker]*info["quantity"]*gfx(info["currency"],sec_vals.index[0],ex_rates)
                                cv_chf  = cur_prices_s[ticker]*info["quantity"]*gfx(info["currency"],sec_vals.index[-1],ex_rates)
                                wt      = iv_chf/sec_vals.iloc[0]*100
                                contrib = (cv_chf-iv_chf)/sec_vals.iloc[0]*100
                                hp.append({"Ticker":ticker,"Name":info["name"],"Return (%)":ret_,
                                           "Weight (%)":wt,"Contribution (%)":contrib})
                        pdf = pd.DataFrame(hp).sort_values("Return (%)", ascending=False)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Best Performers**")
                            st.dataframe(pdf.head(3).style.format({"Return (%)":"{:.2f}%","Weight (%)":"{:.2f}%","Contribution (%)":"{:.2f}%"}), hide_index=True)
                        with c2:
                            st.markdown("**Worst Performers**")
                            st.dataframe(pdf.tail(3).style.format({"Return (%)":"{:.2f}%","Weight (%)":"{:.2f}%","Contribution (%)":"{:.2f}%"}), hide_index=True)
                        st.markdown("**All Holdings**")
                        st.dataframe(pdf.style.format({"Return (%)":"{:.2f}%","Weight (%)":"{:.2f}%","Contribution (%)":"{:.2f}%"})
                                     .background_gradient(subset=["Return (%)"], cmap="RdYlGn"),
                                     hide_index=True, use_container_width=True)

                        # Risk metrics
                        st.subheader("⚠️ Risk Metrics")
                        var95 = np.percentile(sr, 5) * 100
                        if len(sector_holdings) > 1:
                            rdict = {}
                            for ticker, info in sector_holdings.items():
                                if ticker in sector_data:
                                    sd = sector_data[ticker]
                                    sc = sd["Close"].iloc[:,0] if isinstance(sd["Close"],pd.DataFrame) else sd["Close"]
                                    rdict[info["name"]] = sc.pct_change().dropna()
                            corr_m = pd.DataFrame(rdict).corr()
                            c1, c2 = st.columns(2)
                            c1.metric("Value at Risk (95%)", f"{var95:.2f}%")
                            up_tri = corr_m.values[np.triu_indices_from(corr_m.values, k=1)]
                            c2.metric("Average Correlation",  f"{up_tri.mean():.2f}")
                            st.markdown("**Correlation Matrix**")
                            st.dataframe(corr_m.style.background_gradient(cmap="coolwarm",vmin=-1,vmax=1).format("{:.2f}"),
                                         use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())

        # ── Financial Analysis ───────────────────────────────────────────────
        elif sector_tab == "Financial Analysis":
            st.header(f"💰 {sector_name} - Financial Analysis")

            if st.button("📊 Generate Financial Analysis", type="primary", key=f"fin_gen_{sector_name}"):
                try:
                    with st.spinner(f"Fetching financial data…"):
                        fin_rows = []
                        for ticker, info in sector_holdings.items():
                            try:
                                tk = yf.Ticker(ticker); ti = tk.info
                                try:
                                    cf = tk.cash_flow; bs = tk.balance_sheet
                                    ocf_r = None
                                    if not cf.empty and not bs.empty:
                                        ocf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
                                        cl_ = bs.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in bs.index else None
                                        ocf_r = ocf / cl_ if ocf is not None and cl_ and cl_ != 0 else None
                                except Exception: ocf_r = None

                                fin_rows.append({
                                    "Ticker": ticker, "Name": info["name"],
                                    "Market Cap":        ti.get("marketCap"),
                                    "P/E Ratio":         ti.get("trailingPE"),
                                    "Forward P/E":       ti.get("forwardPE"),
                                    "P/B Ratio":         ti.get("priceToBook"),
                                    "Dividend Yield (%)": (ti.get("dividendYield") or 0) * 100,
                                    "Profit Margin (%)":  (ti.get("profitMargins") or 0) * 100,
                                    "ROE (%)":            (ti.get("returnOnEquity") or 0) * 100,
                                    "ROA (%)":            (ti.get("returnOnAssets") or 0) * 100,
                                    "Debt/Equity":        ti.get("debtToEquity"),
                                    "Current Ratio":      ti.get("currentRatio"),
                                    "Revenue Growth (%)": (ti.get("revenueGrowth") or 0) * 100,
                                    "Beta":               ti.get("beta"),
                                    "OCF Ratio":          ocf_r,
                                })
                            except Exception as e:
                                st.warning(f"Could not fetch data for {ticker}: {e}")

                    if fin_rows:
                        fin_df = pd.DataFrame(fin_rows)
                        st.subheader("📊 Key Financial Ratios")
                        # ── Fetch true sector industry average from yfinance ──────────────
                        # Map our sector names to yfinance industry/sector ETF or
                        # use the sector field from yfinance .info to group peers
                        SECTOR_ETFS = {
                            "TMT":           "XLK",
                            "FIG":           "XLF",
                            "Industrials":   "XLI",
                            "PUI":           "XLB",
                            "Consumer Goods":"XLP",
                            "Healthcare":    "XLV",
                            "Real Estate":   "XLRE",
                            "Energy":        "XLE",
                            "Utilities":     "XLU",
                        }
                        # Build industry averages from yfinance info of each holding's sector peers
                        # We use the median of all portfolio companies' yfinance sector peers
                        # fetched via their "industry" group — or fall back to the ETF constituents proxy
                        # For simplicity: fetch ~10 large-cap peers for the sector ETF and average their ratios
                        @st.cache_data(show_spinner=False)
                        def fetch_sector_industry_avg(sector, ratio_cols):
                            """
                            Returns a dict of {ratio_col: industry_average} using yfinance
                            info from the top holdings of the sector ETF as a proxy for
                            the true sector average.
                            """
                            etf_ticker = SECTOR_ETFS.get(sector, "SPY")
                            # Representative large-cap tickers per sector as fallback proxies
                            SECTOR_PROXIES = {
                                "XLK":  ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","AMD","ADBE","CSCO"],
                                "XLF":  ["BRK-B","JPM","V","MA","BAC","GS","MS","WFC","AXP","BLK"],
                                "XLI":  ["GE","CAT","RTX","HON","UNP","BA","LMT","DE","MMM","EMR"],
                                "XLB":  ["LIN","APD","ECL","SHW","FCX","NEM","VMC","MLM","ALB","CE"],
                                "XLP":  ["PG","COST","WMT","KO","PEP","MDLZ","PM","MO","CL","GIS"],
                                "XLV":  ["LLY","UNH","JNJ","MRK","ABBV","TMO","ABT","DHR","PFE","AMGN"],
                                "XLRE": ["AMT","PLD","CCI","EQIX","PSA","O","WELL","DLR","SPG","AVB"],
                                "XLE":  ["XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","OXY"],
                                "XLU":  ["NEE","DUK","SO","D","AEP","EXC","XEL","ED","ETR","WEC"],
                                "SPY":  ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","JPM","V","UNH"],
                            }
                            proxies = SECTOR_PROXIES.get(etf_ticker, SECTOR_PROXIES["SPY"])
                            peer_vals = {col: [] for col in ratio_cols}
                            for pt in proxies:
                                try:
                                    pi = yf.Ticker(pt).info
                                    mapping = {
                                        "P/E Ratio":   pi.get("trailingPE"),
                                        "P/B Ratio":   pi.get("priceToBook"),
                                        "Debt/Equity": pi.get("debtToEquity"),
                                        "OCF Ratio":   None,  # complex, skip for peers
                                        "Forward P/E": pi.get("forwardPE"),
                                        "Profit Margin (%)": (pi.get("profitMargins") or 0)*100,
                                        "ROE (%)": (pi.get("returnOnEquity") or 0)*100,
                                        "ROA (%)": (pi.get("returnOnAssets") or 0)*100,
                                        "Dividend Yield (%)": (pi.get("dividendYield") or 0)*100,
                                        "Revenue Growth (%)": (pi.get("revenueGrowth") or 0)*100,
                                        "Beta": pi.get("beta"),
                                        "Current Ratio": pi.get("currentRatio"),
                                    }
                                    for col in ratio_cols:
                                        v = mapping.get(col)
                                        if v is not None:
                                            try:
                                                fv = float(v)
                                                # Sanity-filter extreme outliers
                                                if col in ["P/E Ratio","Forward P/E"] and (fv < 0 or fv > 200): continue
                                                if col == "Debt/Equity" and fv > 500: continue
                                                peer_vals[col].append(fv)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                            result = {}
                            for col, vals in peer_vals.items():
                                if vals:
                                    import statistics
                                    result[col] = statistics.median(vals)
                            return result

                        with st.spinner("Fetching sector industry benchmarks…"):
                            ind_avgs = fetch_sector_industry_avg(sector_name, 
                                ["P/E Ratio","P/B Ratio","Debt/Equity","OCF Ratio",
                                 "Forward P/E","Profit Margin (%)","ROE (%)","ROA (%)","Beta","Current Ratio"])

                        ratio_cfg = [("P/E Ratio","Price-to-Earnings"),("P/B Ratio","Price-to-Book"),
                                     ("Debt/Equity","Debt-to-Equity"),("OCF Ratio","Operating Cash Flow Ratio")]
                        c1, c2 = st.columns(2)
                        for idx, (col, title) in enumerate(ratio_cfg):
                            cd = fin_df[["Name", col]].dropna()
                            if not cd.empty:
                                # Use true industry median as benchmark line
                                ind_avg = ind_avgs.get(col)
                                port_avg = cd[col].mean()
                                fig = go.Figure()
                                fig.add_trace(go.Bar(x=cd["Name"], y=cd[col], name=title,
                                                     marker_color="#030C30", text=cd[col].round(2), textposition="outside"))
                                if ind_avg is not None:
                                    fig.add_trace(go.Scatter(x=cd["Name"], y=[ind_avg]*len(cd), mode="lines",
                                                             name=f"Industry Median: {ind_avg:.2f}",
                                                             line=dict(color="#FF4B4B", width=2, dash="dot")))
                                fig.add_trace(go.Scatter(x=cd["Name"], y=[port_avg]*len(cd), mode="lines",
                                                         name=f"Our Portfolio Avg: {port_avg:.2f}",
                                                         line=dict(color="#FFA500", width=1.5, dash="dashdot")))
                                fig.update_layout(title=title, height=400, showlegend=True,
                                                  hovermode="x unified", template="plotly_white")
                                (c1 if idx % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)

                        st.subheader("📋 Comprehensive Financial Summary")
                        st.dataframe(fin_df.fillna("N/A"), hide_index=True, use_container_width=True)
                        st.subheader("🔍 Sector Statistics")
                        nd = fin_df.select_dtypes(include=[np.number])
                        if not nd.empty:
                            st.dataframe(nd.describe().T[["mean","50%","min","max","std"]].rename(columns={"50%":"median"}),
                                         use_container_width=True)
                    else:
                        st.warning("No financial data could be retrieved.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # ── Company Specific ─────────────────────────────────────────────────
        elif sector_tab == "Company Specific":
            st.header(f"🏢 {sector_name} - Company Specific Analysis")

            comp_opts = {info["name"]: ticker for ticker, info in sector_holdings.items()}
            sel_name  = st.selectbox("Select Company", list(comp_opts.keys()), key=f"cs_{sector_name}")

            if sel_name:
                sel_ticker = comp_opts[sel_name]
                cinfo      = sector_holdings[sel_ticker]

                st.subheader(f"📊 {sel_name} ({sel_ticker})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sector",        cinfo["sector"])
                c2.metric("Currency",      cinfo["currency"])
                c3.metric("Quantity Held", f"{cinfo['quantity']:,}")
                c4.metric("Avg Buy Price", f"{cinfo['purchase_price']:.2f}")

                # ── Transaction history for this stock ───────────────────────
                st.markdown("---")
                st.subheader("📋 Transaction History for This Stock")
                tx_stock = cinfo.get("_transactions", pd.DataFrame())
                if not tx_stock.empty:
                    tx_display = tx_stock.copy()
                    tx_display["Date"] = tx_display["Date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(tx_display[["Date","Ticker","Action","Quantity"]],
                                 use_container_width=True, hide_index=True)

                    # Mini waterfall of running position
                    tx_stock_sorted = tx_stock.sort_values("Date")
                    running = 0
                    dates_run, vals_run = [], []
                    for _, r in tx_stock_sorted.iterrows():
                        running += r["Quantity"] if r["Action"] == "Buy" else -r["Quantity"]
                        dates_run.append(r["Date"]); vals_run.append(running)
                    fig_tx = go.Figure(go.Scatter(x=dates_run, y=vals_run, mode="lines+markers",
                                                  line=dict(color="#0F1D64", width=2),
                                                  marker=dict(size=8)))
                    fig_tx.update_layout(title="Running Net Position Over Time",
                                         xaxis_title="Date", yaxis_title="Net Shares Held",
                                         height=300, template="plotly_white")
                    st.plotly_chart(fig_tx, use_container_width=True)

                st.markdown("---")
                st.subheader("📈 Stock Price Analysis")
                c1, c2 = st.columns(2)
                pd_dt = pd.to_datetime(cinfo["purchase_date"])
                with c1: chart_start = st.date_input("Chart Start Date", (pd_dt - pd.Timedelta(days=30)).date(), key=f"cs_{sel_ticker}")
                with c2: chart_end   = st.date_input("Chart End Date",   pd.to_datetime("today"),                key=f"ce_{sel_ticker}")

                try:
                    tk_    = yf.Ticker(sel_ticker)
                    tkinfo = tk_.info
                    an_target = tkinfo.get("targetMeanPrice")
                    an_count  = tkinfo.get("numberOfAnalystOpinions", 0)
                    an_rec    = tkinfo.get("recommendationKey", "N/A").upper()
                except Exception:
                    an_target = None; an_count = 0; an_rec = "N/A"

                target_price = cinfo.get("Target_price")

                if st.button("📊 Generate Stock Analysis", type="primary", key=f"gsa_{sel_ticker}"):
                    if chart_start >= chart_end:
                        st.error("Start date must be before end date.")
                    else:
                        try:
                            with st.spinner(f"Fetching {sel_name}…"):
                                sd = yf.download(sel_ticker, start=chart_start, end=chart_end, progress=False)
                            if sd.empty:
                                st.error("No data.")
                            else:
                                cl = sd["Close"].iloc[:,0] if isinstance(sd["Close"],pd.DataFrame) else sd["Close"]
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=cl.index, y=cl, mode="lines",
                                                         name=f"{sel_ticker} Price",
                                                         line=dict(color="#0F1D64",width=2)))

                                # Mark ALL buy/sell transactions on the chart
                                if not tx_stock.empty:
                                    for _, tr in tx_stock.iterrows():
                                        td = tr["Date"]
                                        if chart_start <= td.date() <= chart_end:
                                            color  = "green" if tr["Action"] == "Buy" else "red"
                                            symbol = "triangle-up" if tr["Action"] == "Buy" else "triangle-down"
                                            avail  = cl.index[cl.index <= td]
                                            if len(avail):
                                                price_at = float(cl.loc[avail[-1]])
                                                fig.add_trace(go.Scatter(
                                                    x=[td], y=[price_at], mode="markers",
                                                    marker=dict(color=color, size=12, symbol=symbol),
                                                    name=f"{tr['Action']} {int(tr['Quantity'])} @ {td.strftime('%Y-%m-%d')}",
                                                    showlegend=True,
                                                ))

                                if target_price and target_price > 0:
                                    fig.add_shape(type="line",x0=0,x1=1,xref="paper",
                                                  y0=target_price,y1=target_price,
                                                  line=dict(color="red",width=2,dash="dash"))
                                    fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
                                                             name=f"Target Price: {target_price:.2f}",
                                                             line=dict(color="red",width=2,dash="dash"),showlegend=True))

                                fig.update_layout(title=f"{sel_name} – Stock Price History",
                                                  xaxis_title="Date", yaxis_title=f"Price ({cinfo['currency']})",
                                                  hovermode="x unified", height=520, template="plotly_white",
                                                  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
                                fig.update_xaxes(rangeslider_visible=True,
                                                 rangeselector=dict(buttons=[
                                                     dict(count=1,label="1M",step="month",stepmode="backward"),
                                                     dict(count=3,label="3M",step="month",stepmode="backward"),
                                                     dict(count=6,label="6M",step="month",stepmode="backward"),
                                                     dict(count=1,label="1Y",step="year",stepmode="backward"),
                                                     dict(step="all",label="All"),
                                                 ]))
                                st.plotly_chart(fig, use_container_width=True)

                                cur_p = float(cl.iloc[-1])
                                pp    = cinfo["purchase_price"]
                                tr_   = ((cur_p - pp) / pp * 100) if pp > 0 else 0

                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Current Price",   f"{cur_p:.2f} {cinfo['currency']}", delta=f"{tr_:.2f}%")
                                c2.metric("Position Value",  f"{cur_p*cinfo['quantity']:,.2f} {cinfo['currency']}")
                                c3.metric("Total Gain/Loss", f"{(cur_p-pp)*cinfo['quantity']:,.2f} {cinfo['currency']}", delta=f"{tr_:.2f}%")
                                if target_price and target_price > 0:
                                    c4.metric("Upside to Target", f"{((target_price-cur_p)/cur_p*100):.2f}%",
                                              delta=f"{target_price:.2f} target")
                                else:
                                    c4.metric("Upside to Target", "N/A")

                                st.markdown("---")
                                st.subheader("🎯 Price Targets")
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Your Target Price",   f"{target_price:.2f} {cinfo['currency']}" if target_price else "Not Set")
                                c2.metric("Analyst Consensus",   f"{an_target:.2f} {cinfo['currency']}" if an_target else "N/A")
                                c3.metric("Number of Analysts",  an_count if an_target else "N/A")
                                c4.metric("Recommendation",      an_rec)

                                st.markdown("---")
                                st.subheader("💡 Investment Thesis")
                                st.info(cinfo.get("thesis","No thesis available"))

                                st.markdown("---")
                                st.subheader("💰 DCF Valuation Parameters")
                                c1, c2 = st.columns(2)
                                c1.write(f"**WACC:** {cinfo.get('WACC','N/A')}%")
                                with c2:
                                    st.write("**Cash Flow Projections:**")
                                    for i in range(1, 6):
                                        st.write(f"Year {i}: {cinfo.get(f'CF_{i}','N/A')}")

                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback; st.code(traceback.format_exc())

# =============================================================================
# ADDING TOOL
# =============================================================================
elif main_page == "Adding Tool":
    st.title("🔧 Portfolio Addition Simulator")
    st.markdown("""
### This tool will allow you to:

**Stock Addition Simulation:**
- Input a stock ticker to analyze
- Specify allocation amount or percentage
- Select position size relative to current portfolio

**Impact Analysis:**
- Portfolio Composition Changes (sector, geo, market cap)
- Performance Impact (historical sim, expected return, Sharpe)
- Risk Metrics Changes (Beta, volatility, correlation, diversification)
- Financial Ratios Impact (P/E, P/B, profitability, debt)

**Comparison View:** Side-by-side before/after metrics with visual charts.
""")
    st.info("🛠️ Stock addition simulation tool coming soon.")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL, MSFT, NESN.SW")
        st.number_input("Allocation Amount (CHF)", 0, value=10000, step=1000)
    with c2:
        st.selectbox("Allocation Method", ["Fixed Amount","Percentage of Portfolio","Equal Weight"])
        st.button("Run Simulation", type="primary", disabled=True)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.info("**Portfolio Dashboard v2.0**\n\nTransaction-history based portfolio tracking.")
