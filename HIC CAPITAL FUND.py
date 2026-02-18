import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import yfinance as yf
import json, time
from pathlib import Path

# =============================================================================
# LOAD TRANSACTION HISTORY & STATIC INFO FROM EXCEL
# =============================================================================

# =============================================================================
# CURRENCY CACHE  — persists to disk so a cold-boot rate-limit never poisons values
# =============================================================================
_CCY_CACHE_FILE = Path("currency_cache.json")

def _resolve_currencies(tickers: list[str]) -> dict[str, str]:
    """
    Return {ticker: currency_code} for every ticker in *tickers*.

    Strategy per ticker:
      1. Read from currency_cache.json  →  instant, survives reboots
      2. If not cached (or previously failed): fetch from yfinance with retry
      3. On success: write to cache file immediately so next boot is free
      4. On persistent failure: leave out of cache so the next boot retries again
         (never permanently store a wrong value)
    """
    # Load existing cache from disk
    disk_cache: dict = {}
    if _CCY_CACHE_FILE.exists():
        try:
            disk_cache = json.loads(_CCY_CACHE_FILE.read_text())
        except Exception:
            disk_cache = {}

    result     = {}
    cache_dirty = False

    for ticker in tickers:
        # Already cached and looks valid
        cached = disk_cache.get(ticker)
        if cached and isinstance(cached, str) and len(cached) == 3:
            result[ticker] = cached
            print(f"  {ticker}: {cached}  [disk cache]")
            continue

        # Need to fetch from yfinance — retry up to 4 times with backoff
        ccy = None
        for attempt in range(4):
            try:
                info = yf.Ticker(ticker).fast_info   # fast_info is lighter than .info
                raw_ccy = getattr(info, "currency", None)
                if raw_ccy and isinstance(raw_ccy, str) and len(raw_ccy) == 3:
                    ccy = raw_ccy.upper()
                    break
                # fast_info may not have it — try full .info
                full = yf.Ticker(ticker).info
                raw_ccy = full.get("currency")
                if raw_ccy and isinstance(raw_ccy, str) and len(raw_ccy) == 3:
                    ccy = raw_ccy.upper()
                    break
            except Exception as e:
                wait = 2 ** attempt          # 1 s, 2 s, 4 s, 8 s
                print(f"  {ticker}: yfinance error ({e}), retrying in {wait}s…")
                time.sleep(wait)

        if ccy:
            result[ticker]      = ccy
            disk_cache[ticker]  = ccy
            cache_dirty         = True
            print(f"  {ticker}: {ccy}  [yfinance → cached]")
        else:
            # Don't write a failed entry — next boot will retry
            result[ticker] = "USD"
            print(f"  {ticker}: could not resolve, defaulting to USD (not cached)")

        time.sleep(0.3)   # gentle inter-ticker pause

    # Persist updated cache to disk
    if cache_dirty:
        try:
            _CCY_CACHE_FILE.write_text(json.dumps(disk_cache, indent=2))
        except Exception as e:
            print(f"Warning: could not write currency cache: {e}")

    return result


@st.cache_data(show_spinner=False)
def load_data():
    EXCEL_PATH = "portfolio_holdings.xlsx"
    xl         = pd.ExcelFile(EXCEL_PATH)
    sheet_names = xl.sheet_names
    print(f"Sheets found in Excel: {sheet_names}")

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
                print(f"Found data in sheet: '{sname}'")
                break
        if raw is not None:
            break

    if raw is None:
        raise ValueError(f"Could not find any sheet with Buy/Sell data. Sheets available: {sheet_names}")

    raw.columns = [str(c).strip() for c in raw.columns]
    col_lower   = {c.lower(): c for c in raw.columns}

    date_col   = next((col_lower[k] for k in ["date", "trade date", "tradedate"]              if k in col_lower), raw.columns[0])
    ticker_col = next((col_lower[k] for k in ["ticker", "security", "symbol", "stock"]        if k in col_lower), raw.columns[1])
    action_col = next((col_lower[k] for k in ["action", "type", "side", "buy/sell"]           if k in col_lower), raw.columns[2])
    qty_col    = next((col_lower[k] for k in ["shares", "quantity", "qty", "units", "volume"] if k in col_lower), raw.columns[3])

    print(f"   columns detected → date='{date_col}' ticker='{ticker_col}' action='{action_col}' qty='{qty_col}'")

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
    tx["YF_Ticker"] = tx["Ticker"]

    core_cols = {date_col, ticker_col, action_col, qty_col}
    info_extra_cols = [c for c in raw.columns if c not in core_cols]
    info_col_map = {c.lower(): c for c in info_extra_cols}

    unique_tickers = tx["Ticker"].unique()

    INFO_DEFAULTS = {
        "name":         lambda t: t,
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
        for candidate in FIELD_CANDIDATES[field]:
            if candidate in info_col_map:
                return info_col_map[candidate]
        return None

    # ── Currency resolution via yfinance with disk cache ──────────────────────
    yf_currencies = _resolve_currencies(list(unique_tickers))

    INFO_DEFAULTS["currency"] = lambda t: yf_currencies.get(t, "USD")

    detected = {field: resolve_col(field) for field in FIELD_CANDIDATES}
    print(f"\nStatic columns detected in Excel: {detected}")

    info_records = {}
    for ticker in unique_tickers:
        ticker_rows = raw[raw[ticker_col].astype(str).str.strip() == ticker]
        rec = {}
        for field, default_fn in INFO_DEFAULTS.items():
            col = resolve_col(field)
            if field == "currency":
                rec[field] = yf_currencies[ticker]
                continue
            if col is not None:
                vals = ticker_rows[col].dropna()
                vals = vals[vals.astype(str).str.strip() != ""]
                rec[field] = vals.iloc[0] if len(vals) > 0 else default_fn(ticker)
            else:
                rec[field] = default_fn(ticker)
        info_records[ticker] = rec

    info_df = pd.DataFrame.from_dict(info_records, orient="index")
    info_df.index.name = "Ticker"

    print(f"\nSheet: '{used_sheet}' | Transactions: {len(tx)} rows | Unique tickers: {len(info_df)}")
    print(f"{'Ticker':<20} {'Currency':<8} {'Sector':<20} {'Thesis':<30}")
    print("-" * 80)
    for t in sorted(info_df.index):
        row = info_df.loc[t]
        print(f"{t:<20} {str(row.get('currency','?')):<8} {str(row.get('sector','?')):<20} {str(row.get('thesis',''))[:28]}")

    return tx, info_df


def build_portfolio(tx: pd.DataFrame, info_df: pd.DataFrame) -> dict:
    portfolio = {}

    for yf_ticker, grp in tx.groupby("YF_Ticker"):
        if pd.isna(yf_ticker):
            continue

        buys  = grp[grp["Action"] == "Buy"]
        sells = grp[grp["Action"] == "Sell"]
        net_qty = int(buys["Quantity"].sum()) - int(sells["Quantity"].sum())
        if net_qty <= 0:
            continue

        if yf_ticker in info_df.index:
            row = info_df.loc[yf_ticker]
        else:
            st.warning(f" No static info for **{yf_ticker}** – using defaults.")
            row = pd.Series({
                "name": yf_ticker, "target_price": 0.0, "currency": "USD",
                "sector": "Unknown", "thesis": "", "wacc": "",
                "cf_1": "", "cf_2": "", "cf_3": "", "cf_4": "", "cf_5": "",
            })

        first_buy_date = buys["Date"].min()
        purchase_price = _vwap_purchase_price(yf_ticker, buys)

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
            "_transactions":  grp.reset_index(drop=True),
        }
    return portfolio


def _vwap_purchase_price(ticker: str, buys: pd.DataFrame) -> float:
    """Weighted-average purchase price in NATIVE currency."""
    total_cost = 0.0
    total_qty  = 0
    for _, row in buys.iterrows():
        qty   = int(row["Quantity"])
        price = _fetch_close_on_date(ticker, row["Date"])
        if price is None:
            price = 0.0
        total_cost += price * qty
        total_qty  += qty
    return (total_cost / total_qty) if total_qty > 0 else 0.0


def _fetch_close_on_date(ticker: str, date: pd.Timestamp) -> float | None:
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
print("Loading transaction history…")
_tx, _info_df = load_data()

print("Building portfolio from transactions…")
portfolio_holdings = build_portfolio(_tx, _info_df)

print(f"Active positions: {list(portfolio_holdings.keys())}")

# =============================================================================
# FX UTILITIES  (everything converts TO USD as the fund's base currency)
# =============================================================================

NATIVE_TO_USD_PAIRS = {
    "USD": None,
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "JPYUSD=X",
    "CHF": "CHFUSD=X",
    "CAD": "CADUSD=X",
    "AUD": "AUDUSD=X",
    "HKD": "HKDUSD=X",
    "INR": "INRUSD=X",
    "CNY": "CNYUSD=X",
    "SEK": "SEKUSD=X",
    "NOK": "NOKUSD=X",
    "DKK": "DKKUSD=X",
    "SGD": "SGDUSD=X",
    "KRW": "KRWUSD=X",
    "BRL": "BRLUSD=X",
    "MXN": "MXNUSD=X",
}

FALLBACK_FX_TO_USD = {
    "USD": 1.0000,
    "EUR": 1.0850,
    "GBP": 1.2700,
    "JPY": 0.0067,
    "CHF": 1.1260,
    "CAD": 0.7350,
    "AUD": 0.6500,
    "HKD": 0.1280,
    "INR": 0.0120,
    "CNY": 0.1380,
    "SEK": 0.0940,
    "NOK": 0.0940,
    "DKK": 0.1440,
    "SGD": 0.7450,
    "KRW": 0.00073,
    "BRL": 0.1900,
    "MXN": 0.0580,
}

NATIVE_TO_CHF_PAIRS = {
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
    "CHF": None,
}
FALLBACK_FX_TO_CHF = {
    "USD": 0.888, "EUR": 0.940, "INR": 0.0104, "HKD": 0.114,
    "AUD": 0.570, "CAD": 0.650, "GBP": 1.120,  "JPY": 0.0059,
    "CNY": 0.122, "SGD": 0.660, "SEK": 0.083,  "NOK": 0.083,
    "DKK": 0.126, "KRW": 0.00065, "CHF": 1.0,
}


def _get_yf_pair(currency: str, target: str = "USD") -> str | None:
    if currency == target:
        return None
    if target == "USD":
        return NATIVE_TO_USD_PAIRS.get(currency, f"{currency}USD=X")
    if target == "CHF":
        return NATIVE_TO_CHF_PAIRS.get(currency, f"{currency}CHF=X")
    return f"{currency}{target}=X"


_fx_series_cache: dict = {}


def _fetch_fx_series(currency: str, start, end, target: str = "USD") -> pd.DataFrame | None:
    pair = _get_yf_pair(currency, target)
    if pair is None:
        return None

    key = (pair, str(start), str(end))
    if key not in _fx_series_cache:
        try:
            data = yf.download(pair, start=start, end=end, progress=False)
            _fx_series_cache[key] = data if not data.empty else pd.DataFrame()
        except Exception:
            _fx_series_cache[key] = pd.DataFrame()
    return _fx_series_cache[key]


def _fx_rate_on_date(currency: str, date, fx_series, target: str = "USD") -> float:
    if currency == target:
        return 1.0
    fallbacks = FALLBACK_FX_TO_USD if target == "USD" else FALLBACK_FX_TO_CHF
    if fx_series is None or (hasattr(fx_series, "empty") and fx_series.empty):
        return fallbacks.get(currency, 1.0)
    try:
        close = fx_series["Close"].iloc[:, 0] if isinstance(fx_series["Close"], pd.DataFrame) else fx_series["Close"]
        avail = close.index[close.index <= date]
        return float(close.loc[avail[-1]]) if len(avail) > 0 else float(close.iloc[0])
    except Exception:
        return fallbacks.get(currency, 1.0)


def _spot_fx(currency: str, target: str = "USD") -> float:
    if currency == target:
        return 1.0
    pair = _get_yf_pair(currency, target)
    if pair is None:
        return 1.0
    try:
        d = yf.Ticker(pair).history(period="2d")
        if not d.empty:
            return float(d["Close"].iloc[-1])
    except Exception:
        pass
    fallbacks = FALLBACK_FX_TO_USD if target == "USD" else FALLBACK_FX_TO_CHF
    return fallbacks.get(currency, 1.0)


# =============================================================================
# FINANCIAL ANALYSIS HELPERS
# =============================================================================
import statistics as _stats

_SECTOR_ETFS = {
    "TMT":"XLK","FIG":"XLF","Industrials":"XLI","PUI":"XLB",
    "Consumer Goods":"XLP","Healthcare":"XLV",
    "Real Estate":"XLRE","Energy":"XLE","Utilities":"XLU",
}
_SECTOR_PROXIES = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","AMD","ADBE","CSCO"],
    "XLF": ["BRK-B","JPM","V","MA","BAC","GS","MS","WFC","AXP","BLK"],
    "XLI": ["GE","CAT","RTX","HON","UNP","BA","LMT","DE","MMM","EMR"],
    "XLB": ["LIN","APD","ECL","SHW","FCX","NEM","VMC","MLM","ALB","CE"],
    "XLP": ["PG","COST","WMT","KO","PEP","MDLZ","PM","MO","CL","GIS"],
    "XLV": ["LLY","UNH","JNJ","MRK","ABBV","TMO","ABT","DHR","PFE","AMGN"],
    "XLRE":["AMT","PLD","CCI","EQIX","PSA","O","WELL","DLR","SPG","AVB"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","OXY"],
    "XLU": ["NEE","DUK","SO","D","AEP","EXC","XEL","ED","ETR","WEC"],
    "SPY": ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","JPM","V","UNH"],
}
_ALL_RATIO_COLS = (
    "P/E Ratio","P/B Ratio","Debt/Equity","OCF Ratio",
    "Forward P/E","Profit Margin (%)","ROE (%)","ROA (%)","Beta","Current Ratio",
)


def _yf_info_with_retry(ticker: str, max_attempts: int = 4, base_delay: float = 3.0):
    CORE_FIELDS = ("marketCap", "trailingPE", "priceToBook",
                   "returnOnEquity", "currentPrice", "regularMarketPrice")

    for attempt in range(max_attempts):
        try:
            info = yf.Ticker(ticker).info
            if info and any(info.get(f) is not None for f in CORE_FIELDS):
                return info, None
            raise ValueError(
                f"Rate-limit stub returned — none of {CORE_FIELDS} populated "
                f"({len(info)} keys total)"
            )
        except Exception as e:
            err = str(e)
            if attempt < max_attempts - 1:
                wait = base_delay * (2 ** attempt)
                print(f"  {ticker} attempt {attempt+1} failed: {err} — waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                return {}, err
    return {}, "Max retries exceeded"


def _extract_ratios(ti: dict, cf_df=None, bs_df=None) -> dict:
    ocf_r = None
    if cf_df is not None and bs_df is not None:
        try:
            if not cf_df.empty and not bs_df.empty:
                ocf = cf_df.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf_df.index else None
                cl_ = bs_df.loc["Current Liabilities"].iloc[0]  if "Current Liabilities"  in bs_df.index else None
                ocf_r = float(ocf) / float(cl_) if ocf is not None and cl_ and float(cl_) != 0 else None
        except Exception:
            pass
    return {
        "Market Cap":         ti.get("marketCap"),
        "P/E Ratio":          ti.get("trailingPE"),
        "Forward P/E":        ti.get("forwardPE"),
        "P/B Ratio":          ti.get("priceToBook"),
        "Dividend Yield (%)": (ti.get("dividendYield")   or 0) * 100,
        "Profit Margin (%)":  (ti.get("profitMargins")   or 0) * 100,
        "ROE (%)":            (ti.get("returnOnEquity")  or 0) * 100,
        "ROA (%)":            (ti.get("returnOnAssets")  or 0) * 100,
        "Debt/Equity":        ti.get("debtToEquity"),
        "Current Ratio":      ti.get("currentRatio"),
        "Revenue Growth (%)": (ti.get("revenueGrowth")   or 0) * 100,
        "Beta":               ti.get("beta"),
        "OCF Ratio":          ocf_r,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sector_industry_avg(sector: str) -> dict:
    ratio_cols = list(_ALL_RATIO_COLS)
    etf_ticker = _SECTOR_ETFS.get(sector, "SPY")
    proxies    = _SECTOR_PROXIES.get(etf_ticker, _SECTOR_PROXIES["SPY"])
    peer_vals  = {col: [] for col in ratio_cols}

    for pt in proxies:
        info, err = _yf_info_with_retry(pt, max_attempts=3, base_delay=2.0)
        if err:
            print(f"  peer {pt} failed: {err}")
            continue
        mapping = {
            "P/E Ratio":          info.get("trailingPE"),
            "P/B Ratio":          info.get("priceToBook"),
            "Debt/Equity":        info.get("debtToEquity"),
            "OCF Ratio":          None,
            "Forward P/E":        info.get("forwardPE"),
            "Profit Margin (%)":  (info.get("profitMargins")  or 0) * 100,
            "ROE (%)":            (info.get("returnOnEquity") or 0) * 100,
            "ROA (%)":            (info.get("returnOnAssets") or 0) * 100,
            "Dividend Yield (%)": (info.get("dividendYield")  or 0) * 100,
            "Revenue Growth (%)": (info.get("revenueGrowth")  or 0) * 100,
            "Beta":               info.get("beta"),
            "Current Ratio":      info.get("currentRatio"),
        }
        for col in ratio_cols:
            v = mapping.get(col)
            if v is None:
                continue
            try:
                fv = float(v)
                if col in ("P/E Ratio", "Forward P/E") and (fv < 0 or fv > 200): continue
                if col == "Debt/Equity" and fv > 500: continue
                peer_vals[col].append(fv)
            except Exception:
                pass
        time.sleep(0.5)

    return {col: _stats.median(vals) for col, vals in peer_vals.items() if vals}


# =============================================================================
# PAGE CONFIG & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="",
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
    st.session_state.main_page = "🏠 Home"

if "main_page" not in st.session_state:
    st.session_state.main_page = "Home"
if st.sidebar.button("Transactions", use_container_width=True):
    st.session_state.main_page = "Transactions"

st.sidebar.markdown("---")
st.sidebar.subheader("Sectors")

if st.sidebar.button("📱 TMT",             use_container_width=True): st.session_state.main_page = "📱 TMT Sector"
if st.sidebar.button("🏦 FIG",             use_container_width=True): st.session_state.main_page = "🏦 FIG Sector"
if st.sidebar.button("🏭 Industrials",     use_container_width=True): st.session_state.main_page = "🏭 Industrials Sector"
if st.sidebar.button("⚡ PUI",             use_container_width=True): st.session_state.main_page = "⚡ PUI Sector"
if st.sidebar.button("🛒 Consumer Goods",  use_container_width=True): st.session_state.main_page = "🛒 Consumer Goods Sector"
if st.sidebar.button("🏥 Healthcare",      use_container_width=True): st.session_state.main_page = "🏥 Healthcare Sector"

main_page = st.session_state.main_page

# =============================================================================
# TRANSACTION HISTORY PAGE
# =============================================================================
if main_page == "Transactions":
    st.title("📋 Transaction History")

    @st.cache_data(show_spinner=False)
    def fetch_execution_prices(tx_df: pd.DataFrame) -> pd.Series:
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

    st.markdown("**All Transactions**")

    with st.spinner("Fetching execution prices from yfinance…"):
        exec_prices = fetch_execution_prices(_tx.reset_index(drop=True))

    display_tx = _tx.reset_index(drop=True).copy()
    display_tx["Exec Price"] = exec_prices.values

    def get_ticker_currency(ticker):
        if ticker in portfolio_holdings:
            return portfolio_holdings[ticker]["currency"]
        if ticker in _info_df.index:
            return str(_info_df.loc[ticker, "currency"])
        return ""

    display_tx["Currency"] = display_tx["Ticker"].apply(get_ticker_currency)
    display_tx["Gross Value"] = display_tx.apply(
        lambda r: r["Exec Price"] * r["Quantity"] if pd.notna(r["Exec Price"]) else None, axis=1
    )
    display_tx["Date"]        = display_tx["Date"].dt.strftime("%Y-%m-%d")
    display_tx["Exec Price"]  = display_tx["Exec Price"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    display_tx["Gross Value"] = display_tx["Gross Value"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

    st.dataframe(
        display_tx[["Date", "Ticker", "Action", "Quantity", "Currency", "Exec Price", "Gross Value"]],
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.markdown("**Net Position Summary**")
    rows = []
    for yf_ticker, grp in _tx.groupby("YF_Ticker"):
        if pd.isna(yf_ticker):
            continue
        bought = int(grp[grp["Action"] == "Buy"]["Quantity"].sum())
        sold   = int(grp[grp["Action"] == "Sell"]["Quantity"].sum())
        net    = bought - sold
        rows.append({
            "YF Ticker": yf_ticker, "Bought": bought, "Sold": sold,
            "Net Position": net, "Status": "🟢 Open" if net > 0 else "🔴 Closed",
        })
    summary_df = pd.DataFrame(rows).sort_values("YF Ticker")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Transaction Timeline**")

    with st.spinner("Computing bubble sizes from trade values…"):
        raw_prices = fetch_execution_prices(_tx.reset_index(drop=True))

    bubble_tx = _tx.reset_index(drop=True).copy()
    bubble_tx["exec_price"] = raw_prices.values

    spot_usd_rates = {
        t: _spot_fx(portfolio_holdings.get(t, {}).get("currency", "USD"), "USD")
        for t in bubble_tx["Ticker"].unique()
    }

    bubble_tx["exec_price_usd"] = bubble_tx.apply(
        lambda r: r["exec_price"] * spot_usd_rates.get(r["Ticker"], 1.0)
        if pd.notna(r["exec_price"]) else None, axis=1
    )
    bubble_tx["gross_value"] = bubble_tx.apply(
        lambda r: r["exec_price_usd"] * r["Quantity"] if pd.notna(r["exec_price_usd"]) else r["Quantity"],
        axis=1
    )

    global_max_val = bubble_tx["gross_value"].max()
    bubble_tx["bubble_size"] = (bubble_tx["gross_value"] / global_max_val) * 42 + 8

    fig = go.Figure()
    colors = {"Buy": "#0F1D64", "Sell": "#FF6B6B"}
    for action in ["Buy", "Sell"]:
        subset = bubble_tx[bubble_tx["Action"] == action]
        fig.add_trace(go.Scatter(
            x=subset["Date"], y=subset["Ticker"],
            mode="markers",
            marker=dict(color=colors[action], size=subset["bubble_size"], opacity=0.8, sizemode="diameter"),
            name=action,
            hovertemplate=(
                "<b>%{y}</b><br>Date: %{x}<br>"
                "Qty: %{customdata[0]:,}<br>Value (USD): $%{customdata[1]:,.0f}<extra></extra>"
            ),
            customdata=subset[["Quantity", "gross_value"]].values,
        ))
    fig.update_layout(
        title="Buy & Sell Events (bubble size = trade value in USD, all currencies converted)",
        xaxis_title="Date", yaxis_title="Ticker", height=520, template="plotly_white",
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
        if st.button("Generic Summary\n\nKey metrics, performance vs MSCI World, portfolio treemap",
                     key="gen_summary", use_container_width=True):
            st.session_state.home_tab = "Generic Summary"
    with col2:
        if st.button("Portfolio Structure\n\nSector, geographical, and asset distribution",
                     key="portfolio_struct", use_container_width=True):
            st.session_state.home_tab = "Portfolio Structure Analysis"
    with col3:
        if st.button("Forecast\n\nMonte Carlo, analyst targets, DCF analysis",
                     key="forecast", use_container_width=True):
            st.session_state.home_tab = "Forecast"

    st.markdown("---")
    home_tab = st.session_state.home_tab

    # =========================================================================
    # HOME - Generic Summary
    # =========================================================================
    if home_tab == "Generic Summary":
        st.markdown("## **Generic Summary**")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2025-11-06"),
                                       help="Select the start date for analysis")
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("today"),
                                     help="Select the end date for analysis")

        if st.button("Generate Analysis", type="primary"):
            if start_date >= end_date:
                st.error("Start date must be before end date.")
            else:
                try:
                    with st.spinner("Fetching market data and FX rates…"):
                        msci_world = yf.download("URTH", start=start_date, end=end_date, progress=False)
                        msci_close = (msci_world["Close"].iloc[:, 0]
                                      if isinstance(msci_world["Close"], pd.DataFrame)
                                      else msci_world["Close"])

                        used_currencies = {info["currency"] for info in portfolio_holdings.values()}
                        fx_series_usd: dict = {}
                        for ccy in used_currencies:
                            fx_series_usd[ccy] = _fetch_fx_series(ccy, start_date, end_date, "USD")

                        portfolio_data: dict = {}
                        for ticker in portfolio_holdings:
                            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                            if not data.empty:
                                portfolio_data[ticker] = data

                    def get_price_native(ticker, date, data_dict):
                        if ticker not in data_dict:
                            return None
                        d  = data_dict[ticker]
                        cl = d["Close"].iloc[:, 0] if isinstance(d["Close"], pd.DataFrame) else d["Close"]
                        av = cl.index[cl.index <= date]
                        return float(cl.loc[av[-1]]) if len(av) > 0 else None

                    def get_rate_usd(currency, date):
                        return _fx_rate_on_date(currency, date, fx_series_usd.get(currency), "USD")

                    # ── Risk Metrics ──────────────────────────────────────────
                    st.markdown("### **Portfolio Risk Metrics**")

                    with st.spinner("Calculating Sharpe, Alpha & Beta…"):
                        all_dates = msci_close.index

                        first_trade_date = pd.Timestamp(_tx["Date"].min()).normalize()
                        active_dates = all_dates[all_dates >= first_trade_date]

                        if len(active_dates) < 10:
                            st.warning("Not enough trading days after the first trade to compute reliable risk metrics.")
                        else:
                            portfolio_values = pd.Series(index=active_dates, dtype=float)

                            for date in active_dates:
                                daily_value = 0.0
                                for ticker, info in portfolio_holdings.items():
                                    price_native = get_price_native(ticker, date, portfolio_data)
                                    if price_native is None:
                                        continue
                                    rate = get_rate_usd(info["currency"], date)
                                    daily_value += price_native * info["quantity"] * rate
                                portfolio_values[date] = daily_value

                            first_nonzero = portfolio_values[portfolio_values > 0].index.min()
                            portfolio_values = portfolio_values.loc[first_nonzero:]

                            port_ret = portfolio_values.pct_change().dropna()

                            bad_ret = port_ret[port_ret.abs() > 0.25]
                            if not bad_ret.empty:
                                st.warning(
                                    f"⚠️ {len(bad_ret)} daily return(s) exceed ±25% "
                                    f"({bad_ret.index[0].date()} … {bad_ret.index[-1].date()}). "
                                    "These may indicate FX gaps or data issues and are excluded from risk metrics."
                                )
                                port_ret = port_ret[port_ret.abs() <= 0.25]

                            msci_ret = msci_close.pct_change().dropna()
                            common   = port_ret.index.intersection(msci_ret.index)

                            if len(common) < 10:
                                st.warning("Too few overlapping trading days to compute reliable metrics.")
                            else:
                                pr = port_ret.loc[common]
                                mr = msci_ret.loc[common]

                                rf_daily = 0.02 / 252
                                excess   = pr - rf_daily
                                sharpe   = (excess.mean() / excess.std(ddof=1)) * np.sqrt(252) \
                                           if excess.std(ddof=1) != 0 else 0

                                port_vol_ann = pr.std(ddof=1) * np.sqrt(252) * 100
                                msci_vol_ann = mr.std(ddof=1) * np.sqrt(252) * 100

                                from scipy import stats as scipy_stats
                                reg_data = pd.DataFrame({"msci": mr, "portfolio": pr}).dropna()
                                if len(reg_data) > 2:
                                    slope, intercept, r_val, *_ = scipy_stats.linregress(
                                        reg_data["msci"], reg_data["portfolio"]
                                    )
                                    beta = slope
                                    alpha_annual = intercept * 252
                                    r_squared    = r_val ** 2
                                else:
                                    beta = alpha_annual = r_squared = 0.0

                                n_days = len(pr)
                                st.caption(
                                    f"Computed over **{n_days} trading days** "
                                    f"({pr.index[0].date()} → {pr.index[-1].date()}) "
                                    f"starting from first trade. Risk-free rate: 2% p.a."
                                )

                                c1, c2, c3 = st.columns(3)
                                c1.metric("Sharpe Ratio",         f"{sharpe:.2f}")
                                c2.metric("Beta (vs MSCI World)", f"{beta:.2f}")
                                c3.metric("Alpha (Annualized)",   f"{alpha_annual*100:.2f}%")

                                c1, c2, c3 = st.columns(3)
                                c1.metric("Portfolio Vol (Ann.)", f"{port_vol_ann:.1f}%")
                                c2.metric("MSCI World Vol (Ann.)", f"{msci_vol_ann:.1f}%")
                                c3.metric("R² (vs MSCI World)",   f"{r_squared:.2f}")

                                def _sharpe_label(s):
                                    if   s > 1.5: return "Good"
                                    elif s > 0.5: return "Acceptable"
                                    elif s > 0:   return "Poor"
                                    else:         return "Negative"

                                st.info(f"""
**Interpretation ({n_days} days of live portfolio data):**
- **Sharpe Ratio ({sharpe:.2f})**: {_sharpe_label(sharpe)} risk-adjusted return (>1 is good; >2 is rare and typically only seen over short bull runs)
- **Beta ({beta:.2f})**: Portfolio is {'more volatile than' if beta > 1 else 'less volatile than' if beta < 1 else 'as volatile as'} the MSCI World
- **Alpha ({alpha_annual*100:.2f}% p.a.)**: {'Outperforming' if alpha_annual > 0 else 'Underperforming'} the benchmark on a risk-adjusted basis
- **R² ({r_squared:.2f})**: {r_squared*100:.0f}% of portfolio variance explained by MSCI World moves
""")

                    st.markdown("---")

                    FUND_SIZE_USD = 1_000_000.0

                    all_tx = _tx.sort_values("Date").reset_index(drop=True)
                    tx_by_date: dict = {}
                    for _, tx_row in all_tx.iterrows():
                        d = tx_row["Date"].normalize()
                        tx_by_date.setdefault(d, []).append(tx_row)

                    dates = msci_close.index

                    nav_usd_list    = []
                    equity_usd_list = []
                    cash_usd_list   = []

                    cash_usd   = FUND_SIZE_USD
                    live_shares: dict = {t: 0 for t in portfolio_holdings}
                    for t in all_tx["Ticker"].unique():
                        if t not in live_shares:
                            live_shares[t] = 0

                    for date in dates:
                        date_norm = pd.Timestamp(date).normalize()

                        for tx_row in tx_by_date.get(date_norm, []):
                            ticker = tx_row["Ticker"]
                            qty    = int(tx_row["Quantity"])
                            action = tx_row["Action"]

                            price_native = get_price_native(ticker, date, portfolio_data)
                            if price_native is None:
                                price_native = portfolio_holdings.get(ticker, {}).get("purchase_price", 0.0)

                            ccy = portfolio_holdings.get(ticker, {}).get("currency", "USD")
                            price_usd = price_native * get_rate_usd(ccy, date)

                            if action == "Buy":
                                cash_usd -= price_usd * qty
                                live_shares[ticker] = live_shares.get(ticker, 0) + qty
                            elif action == "Sell":
                                cash_usd += price_usd * qty
                                live_shares[ticker] = max(0, live_shares.get(ticker, 0) - qty)

                        equity_usd = 0.0
                        for ticker, shares in live_shares.items():
                            if shares <= 0:
                                continue
                            price_native = get_price_native(ticker, date, portfolio_data)
                            if price_native is None:
                                continue
                            ccy       = portfolio_holdings.get(ticker, {}).get("currency", "USD")
                            price_usd = price_native * get_rate_usd(ccy, date)
                            equity_usd += price_usd * shares

                        nav_usd_list.append(cash_usd + equity_usd)
                        equity_usd_list.append(equity_usd)
                        cash_usd_list.append(cash_usd)

                    nav_series    = pd.Series(nav_usd_list,    index=dates)
                    equity_series = pd.Series(equity_usd_list, index=dates)
                    cash_series   = pd.Series(cash_usd_list,   index=dates)

                    # ── Performance chart ─────────────────────────────────────
                    st.markdown("### **Portfolio vs MSCI World Performance**")

                    port_norm = nav_series / FUND_SIZE_USD * 100
                    msci_norm = msci_close / msci_close.iloc[0] * 100

                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(
                        x=dates, y=port_norm, mode="lines",
                        name="Total NAV (equity + cash, USD)",
                        line=dict(color="#0F1D64", width=3),
                        hovertemplate="<b>Total NAV</b><br>%{x}<br>$%{customdata:,.0f}<extra></extra>",
                        customdata=nav_series.values,
                    ))
                    fig_perf.add_trace(go.Scatter(
                        x=dates, y=msci_norm, mode="lines",
                        name="MSCI World (URTH)",
                        line=dict(color="#FF6B6B", width=2),
                        hovertemplate="<b>MSCI World</b><br>%{x}<br>Index: %{y:.2f}<extra></extra>",
                    ))
                    fig_perf.update_layout(
                        title="Fund NAV vs MSCI World (Base 100 = $1,000,000 USD) — all FX converted to USD",
                        xaxis_title="Date", yaxis_title="Index Value (Base 100)",
                        hovermode="x unified", height=520, template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)

                    # ── Key metrics ───────────────────────────────────────────
                    final_nav    = nav_series.iloc[-1]
                    final_cash   = cash_series.iloc[-1]
                    final_equity = equity_series.iloc[-1]
                    port_ret_total = (final_nav - FUND_SIZE_USD) / FUND_SIZE_USD * 100
                    msci_ret_total = (msci_close.iloc[-1] - msci_close.iloc[0]) / msci_close.iloc[0] * 100
                    outperf        = port_ret_total - msci_ret_total
                    cash_pct       = final_cash / final_nav * 100 if final_nav > 0 else 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Fund NAV (USD)",    f"${final_nav:,.0f}",      delta=f"{port_ret_total:.2f}% vs $1M")
                    c2.metric("Equity Value (USD)", f"${final_equity:,.0f}",  delta=f"{final_equity/final_nav*100:.1f}% of NAV")
                    c3.metric("Cash (USD)",         f"${final_cash:,.0f}",    delta=f"{cash_pct:.1f}% of NAV")
                    c4.metric("vs MSCI World",      f"{outperf:+.2f}%",       delta=f"{outperf:+.2f}%")

                    st.info(f"""
**Fund Summary (all amounts in USD):** Initial capital **$1,000,000** | 
Fund return **{port_ret_total:+.2f}%** | MSCI World **{msci_ret_total:+.2f}%** | 
Alpha **{outperf:+.2f}%** | Cash drag **{cash_pct:.1f}%** of NAV  
*Non-USD positions (e.g. INR, HKD) are converted to USD using live FX rates.*
""")

                    # ── Treemap ───────────────────────────────────────────────
                    st.markdown("### **Portfolio Composition Treemap** (USD)")
                    treemap_data = []
                    final_date   = dates[-1]

                    for ticker, shares in live_shares.items():
                        if shares <= 0:
                            continue
                        info = portfolio_holdings.get(ticker)
                        if info is None:
                            continue
                        price_native = get_price_native(ticker, final_date, portfolio_data)
                        if price_native is None:
                            continue
                        ccy       = info["currency"]
                        price_usd = price_native * get_rate_usd(ccy, final_date)
                        curr_val  = price_usd * shares

                        buy_tx   = all_tx[(all_tx["Ticker"] == ticker) & (all_tx["Action"] == "Buy")]
                        cost_usd = 0.0
                        for _, brow in buy_tx.iterrows():
                            bp = get_price_native(ticker, brow["Date"], portfolio_data)
                            if bp is None:
                                bp = info["purchase_price"]
                            bp_usd    = bp * _fx_rate_on_date(ccy, brow["Date"], fx_series_usd.get(ccy), "USD")
                            cost_usd += bp_usd * int(brow["Quantity"])

                        avg_cost_usd = cost_usd / shares if shares > 0 else 0
                        perf = (price_usd - avg_cost_usd) / avg_cost_usd * 100 if avg_cost_usd > 0 else 0

                        treemap_data.append({
                            "ticker": ticker, "name": info.get("name", ticker),
                            "weight": curr_val / final_nav * 100,
                            "performance": perf, "value": curr_val,
                            "currency": ccy,
                            "price_native": price_native,
                            "price_usd": price_usd,
                        })

                    treemap_data.append({
                        "ticker": "CASH", "name": "Cash (USD)",
                        "weight": final_cash / final_nav * 100,
                        "performance": 0.0, "value": final_cash,
                        "currency": "USD", "price_native": 1.0, "price_usd": 1.0,
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
                        hovertemplate="<b>%{label}</b><br>Weight: %{customdata[0]:.2f}%<br>Return: %{customdata[1]:.2f}%<br>Value (USD): $%{customdata[2]:,.0f}<br>Ccy: %{customdata[3]}<extra></extra>",
                    )
                    fig_tm.update_layout(height=600, margin=dict(t=50, l=0, r=0, b=0))
                    st.plotly_chart(fig_tm, use_container_width=True)

                    # ── Holdings table ────────────────────────────────────────
                    st.markdown("### **Holdings Details** (USD, all FX converted)")
                    ht = treemap_df.copy()
                    ht["Return (%)"]   = ht["performance"].apply(lambda x: f"{x:.2f}%")
                    ht["Weight (%)"]   = ht["weight"].apply(lambda x: f"{x:.2f}%")
                    ht["Value (USD)"]  = ht["value"].apply(lambda x: f"${x:,.0f}")
                    ht["Native Price"] = ht.apply(
                        lambda r: f"{r['price_native']:.2f} {r['currency']}" if r["ticker"] != "CASH" else "—",
                        axis=1
                    )
                    ht["USD Price"]    = ht.apply(
                        lambda r: f"${r['price_usd']:.4f}" if r["ticker"] != "CASH" else "—",
                        axis=1
                    )
                    ht = ht[["name","ticker","currency","Native Price","USD Price","Weight (%)","Return (%)","Value (USD)"]]
                    ht.columns = ["Company","Ticker","Native Ccy","Native Price","USD Price","Weight","Return","Value (USD)"]
                    st.dataframe(ht.sort_values("Return", ascending=False),
                                 use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    import traceback; st.code(traceback.format_exc())

    # =========================================================================
    # HOME - Portfolio Structure Analysis
    # =========================================================================
    elif home_tab == "Portfolio Structure Analysis":
        st.markdown("## **Portfolio Structure Analysis**")

        @st.cache_data(show_spinner=False)
        def fetch_hq_countries(tickers: tuple) -> dict:
            EXCHANGE_COUNTRY_FALLBACK = {
                ".NS": "India", ".BO": "India",
                ".DE": "Germany", ".F": "Germany",
                ".SW": "Switzerland", ".PA": "France",
                ".AS": "Netherlands", ".L": "United Kingdom",
                ".HK": "Hong Kong", ".AX": "Australia",
                ".TO": "Canada", ".V": "Canada",
                ".T": "Japan", ".KS": "South Korea",
                ".SS": "China", ".SZ": "China",
                ".BR": "Belgium", ".ST": "Sweden",
                ".CO": "Denmark", ".OL": "Norway",
                ".HE": "Finland", ".LS": "Portugal",
                ".MC": "Spain", ".MI": "Italy",
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
                for suffix, country in EXCHANGE_COUNTRY_FALLBACK.items():
                    if ticker.upper().endswith(suffix.upper()):
                        result[ticker] = country
                        break
                else:
                    result[ticker] = "United States"
            return result

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
                spot_usd_rates = {}
                for ccy in {info["currency"] for info in portfolio_holdings.values()}:
                    spot_usd_rates[ccy] = _spot_fx(ccy, "USD")

                cur_prices = {}
                co_info    = {}
                for ticker, info in portfolio_holdings.items():
                    price = None
                    for _attempt in range(3):
                        try:
                            hist = yf.Ticker(ticker).history(period="2d")
                            if not hist.empty:
                                price = float(hist["Close"].iloc[-1])
                                break
                        except Exception:
                            time.sleep(2 ** _attempt)
                    cur_prices[ticker] = price if price is not None else info["purchase_price"]

                    mc = 0
                    for _attempt in range(3):
                        try:
                            fi_mc = getattr(yf.Ticker(ticker).fast_info, "market_cap", None)
                            if fi_mc and fi_mc > 0:
                                mc = fi_mc
                                break
                            full_mc = yf.Ticker(ticker).info.get("marketCap") or 0
                            if full_mc > 0:
                                mc = full_mc
                                break
                        except Exception:
                            time.sleep(2 ** _attempt)
                    co_info[ticker] = {"market_cap": mc}

            def classify_mc(mc):
                if mc == 0:      return "Unknown"
                elif mc >= 10e9: return "Large"
                elif mc >= 2e9:  return "Mid"
                else:            return "Small"

            holdings_analysis = []
            total_equity_usd  = 0.0
            for ticker, info in portfolio_holdings.items():
                if ticker in cur_prices:
                    fx_to_usd = spot_usd_rates.get(info["currency"], FALLBACK_FX_TO_USD.get(info["currency"], 1.0))
                    val_usd   = cur_prices[ticker] * info["quantity"] * fx_to_usd
                    total_equity_usd += val_usd
                    country = country_mapping.get(ticker) or "United States"
                    holdings_analysis.append({
                        "ticker":      ticker,
                        "name":        info["name"],
                        "sector":      info["sector"],
                        "country":     country,
                        "country_iso": country_iso.get(country, ""),
                        "region":      region_mapping.get(country, "Unknown"),
                        "market_cap":  classify_mc(co_info.get(ticker, {}).get("market_cap", 0)),
                        "currency":    info["currency"],
                        "value_usd":   val_usd,
                        "weight_nav":  0.0,
                        "weight_eq":   0.0,
                    })

            FUND_SIZE_USD = 1_000_000.0
            cash_usd      = FUND_SIZE_USD
            all_tx_sorted = _tx.sort_values("Date").reset_index(drop=True)

            tx_start = all_tx_sorted["Date"].min()
            tx_end   = pd.Timestamp.today()
            used_ccys = {portfolio_holdings.get(t, {}).get("currency", "USD")
                         for t in all_tx_sorted["Ticker"].unique()}
            fx_series_struct = {
                ccy: _fetch_fx_series(ccy, tx_start, tx_end, "USD")
                for ccy in used_ccys
            }

            for _, tx_row in all_tx_sorted.iterrows():
                t          = tx_row["Ticker"]
                qty        = int(tx_row["Quantity"])
                action     = tx_row["Action"]
                trade_date = tx_row["Date"]
                ccy        = portfolio_holdings.get(t, {}).get("currency", "USD")

                price_native = _fetch_close_on_date(t, trade_date)
                if price_native is None:
                    price_native = portfolio_holdings.get(t, {}).get("purchase_price", 0.0)

                fx_rate   = _fx_rate_on_date(ccy, trade_date, fx_series_struct.get(ccy), "USD")
                price_usd = price_native * fx_rate

                if action == "Buy":
                    cash_usd -= price_usd * qty
                elif action == "Sell":
                    cash_usd += price_usd * qty

            cash_usd = max(cash_usd, 0.0)

            total_nav_usd = total_equity_usd + cash_usd

            for h in holdings_analysis:
                h["weight_nav"] = h["value_usd"] / total_nav_usd  * 100 if total_nav_usd  > 0 else 0
                h["weight_eq"]  = h["value_usd"] / total_equity_usd * 100 if total_equity_usd > 0 else 0

            for h in holdings_analysis:
                h["weight"] = h["weight_nav"]

            df_an = pd.DataFrame(holdings_analysis)
            blue  = ["#0F1D64","#1E3A8A","#3B82F6","#60A5FA","#93C5FD","#DBEAFE"]

            cash_pct   = cash_usd / total_nav_usd * 100 if total_nav_usd > 0 else 0
            equity_pct = total_equity_usd / total_nav_usd * 100 if total_nav_usd > 0 else 0
            st.info(
                f"**Fund NAV: ${total_nav_usd:,.0f} USD** — "
                f"Equity: ${total_equity_usd:,.0f} ({equity_pct:.1f}%) · "
                f"Cash: ${cash_usd:,.0f} ({cash_pct:.1f}%)\n"
                f"*Weights below are vs full NAV (equity + cash), consistent with Generic Summary.*"
            )

            st.markdown("### **Sector Distribution**")
            sec_data = df_an.groupby("sector")["weight"].sum().reset_index()
            sec_data.columns = ["Sector","Weight (%)"]
            if cash_pct > 0:
                sec_data = pd.concat([
                    sec_data,
                    pd.DataFrame([{"Sector": "Cash (USD)", "Weight (%)": cash_pct}])
                ], ignore_index=True)

            c1, c2 = st.columns([2,1])
            with c1:
                fig_s = px.pie(sec_data, values="Weight (%)", names="Sector",
                               title="Portfolio Allocation by Sector (% of NAV, USD)",
                               color_discrete_sequence=blue, hole=0.4)
                fig_s.update_traces(textposition="inside", textinfo="percent+label")
                fig_s.update_layout(height=400)
                st.plotly_chart(fig_s, use_container_width=True)
            with c2:
                st.markdown("### Sector Breakdown")
                for _, r in sec_data.sort_values("Weight (%)", ascending=False).iterrows():
                    st.metric(r["Sector"], f"{r['Weight (%)']:.1f}%")

            st.markdown("### **Geographical Distribution**")
            c1, c2 = st.columns([2,1])
            with c1:
                cnt_alloc = df_an.groupby(["country","country_iso"])["weight"].sum().reset_index()
                cnt_alloc.columns = ["Country","ISO","Weight (%)"]
                fig_map = px.choropleth(cnt_alloc, locations="ISO", color="Weight (%)",
                                        hover_name="Country",
                                        hover_data={"ISO":False,"Weight (%)":":.2f"},
                                        color_continuous_scale=[[0,"#DBEAFE"],[0.25,"#93C5FD"],
                                                                 [0.5,"#60A5FA"],[0.75,"#3B82F6"],[1,"#0F1D64"]],
                                        title="Geographic Distribution (% of NAV, USD)")
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

            st.markdown("### **Additional Analysis**")
            c1, c2 = st.columns(2)
            with c1:
                mc_alloc = df_an.groupby("market_cap")["weight"].sum().reset_index()
                mc_alloc.columns = ["Market Cap","Weight (%)"]
                fig_mc = px.bar(mc_alloc, x="Market Cap", y="Weight (%)",
                                title="Market Cap Distribution (% of NAV)", color="Weight (%)",
                                color_continuous_scale=[[0,"#DBEAFE"],[0.5,"#3B82F6"],[1,"#0F1D64"]])
                fig_mc.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)
            with c2:
                cur_data = df_an.groupby("currency")["weight"].sum().reset_index()
                cur_data.columns = ["Currency","Weight (%)"]
                existing_usd = cur_data.loc[cur_data["Currency"]=="USD","Weight (%)"].sum()
                if existing_usd > 0:
                    cur_data.loc[cur_data["Currency"]=="USD","Weight (%)"] += cash_pct
                else:
                    cur_data = pd.concat([
                        cur_data,
                        pd.DataFrame([{"Currency":"USD (cash)","Weight (%)": cash_pct}])
                    ], ignore_index=True)
                fig_cur = px.pie(cur_data, values="Weight (%)", names="Currency",
                                 title="Currency Exposure incl. Cash (% of NAV)",
                                 color_discrete_sequence=blue)
                fig_cur.update_traces(textposition="inside", textinfo="percent+label")
                fig_cur.update_layout(height=350)
                st.plotly_chart(fig_cur, use_container_width=True)

            st.markdown("### **Concentration Metrics**")
            df_sorted  = df_an.sort_values("weight", ascending=False)
            top5_conc  = df_sorted.head(5)["weight"].sum()
            top10_conc = df_sorted.head(10)["weight"].sum()
            hhi        = (df_an["weight"] ** 2).sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Top 5 Holdings",  f"{top5_conc:.1f}% of NAV")
            c2.metric("Top 10 Holdings", f"{top10_conc:.1f}% of NAV")
            with c3:
                st.metric("HHI Index (equity only)", f"{hhi:.0f}")
                st.caption("✅ Well diversified" if hhi < 1000 else "⚠️ Moderately concentrated" if hhi < 1800 else "🔴 Highly concentrated")

            st.markdown("### Top 10 Holdings by Weight (% of NAV)")
            th = df_sorted[["name","sector","country","currency","value_usd","weight"]].head(10).copy()
            th["value_usd"] = th["value_usd"].apply(lambda x: f"${x:,.0f}")
            th["weight"]    = th["weight"].apply(lambda x: f"{x:.2f}%")
            th.columns = ["Company","Sector","Country","Native Ccy","Value (USD)","Weight (% NAV)"]
            st.dataframe(th, use_container_width=True, hide_index=True)

            st.markdown("### **Portfolio Summary** (USD)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Holdings",  len(df_an))
            c2.metric("Fund NAV (USD)",  f"${total_nav_usd:,.0f}")
            c3.metric("Sectors Covered", df_an["sector"].nunique())
            c4.metric("Countries",       df_an["country"].nunique())

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback; st.code(traceback.format_exc())

    # =========================================================================
    # HOME - Forecast
    # =========================================================================
    elif home_tab == "Forecast":
        st.markdown("## **Forecast**")

        with st.spinner("Loading stock data…"):
            stock_data = {}
            for ticker in portfolio_holdings:
                try:
                    hist = yf.Ticker(ticker).history(period="1y")
                    if not hist.empty:
                        stock_data[ticker] = {
                            "current_price": hist["Close"].iloc[-1],
                            "historical": hist,
                        }
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")

        forecast_method = st.tabs(["Monte Carlo Simulation", "Analyst Consensus", "DCF Analysis"])

        # ── Monte Carlo ──────────────────────────────────────────────────────
        with forecast_method[0]:
            st.markdown("### **Monte Carlo Simulation**")
            st.markdown("Probabilistic portfolio performance projections (values in USD, all FX converted)")

            c1, c2, c3 = st.columns(3)
            with c1: num_sim = st.number_input("Simulations",    100, 10000, 1000, 100)
            with c2: time_h  = st.number_input("Horizon (days)",  30,  1825,  252,  30)
            with c3:
                tot_val = sum(
                    stock_data[t]["current_price"]
                    * portfolio_holdings[t]["quantity"]
                    * _spot_fx(portfolio_holdings[t]["currency"], "USD")
                    for t in portfolio_holdings if t in stock_data
                )
                init_inv = st.number_input("Initial Portfolio Value (USD)", 1000, value=int(tot_val), step=1000)

            if st.button("Run Monte Carlo Simulation", type="primary"):
                with st.spinner("Running simulations…"):
                    pv = {
                        t: stock_data[t]["current_price"]
                           * portfolio_holdings[t]["quantity"]
                           * _spot_fx(portfolio_holdings[t]["currency"], "USD")
                        for t in portfolio_holdings if t in stock_data
                    }
                    tot = sum(pv.values())
                    wts = {t: pv[t] / tot for t in pv}
                    ret_data = {
                        t: stock_data[t]["historical"]["Close"].pct_change().dropna()
                        for t in portfolio_holdings if t in stock_data
                    }
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
                        fig.update_layout(
                            title=f"Monte Carlo: {num_sim} scenarios over {time_h} days (USD)",
                            xaxis_title="Days", yaxis_title="Portfolio Value (USD)",
                            height=500, hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        c1, c2, c3, c4 = st.columns(4)
                        for col, val, label in [(c1,p5[-1],"Bear (5th)"),(c2,p50[-1],"Base (50th)"),
                                                (c3,p95[-1],"Bull (95th)"),(c4,np.mean(sims[-1,:]),"Expected (Mean)")]:
                            col.metric(label, f"${val:,.0f}", f"{(val-init_inv)/init_inv*100:.1f}%")

        # ── Analyst Consensus ────────────────────────────────────────────────
        with forecast_method[1]:
            st.markdown("### **Analyst Consensus Forecast**")
            st.caption("Target prices are in native currency; values converted to USD for comparison.")
            analyst_data = []
            for ticker in portfolio_holdings:
                if ticker not in stock_data:
                    continue
                try:
                    info_    = yf.Ticker(ticker).info
                    ccy      = portfolio_holdings[ticker]["currency"]
                    fx_spot  = _spot_fx(ccy, "USD")
                    cp_native = stock_data[ticker]["current_price"]
                    cp_usd   = cp_native * fx_spot
                    tp_native = info_.get("targetMeanPrice")
                    if tp_native:
                        tp_usd = tp_native * fx_spot
                        up     = (tp_usd - cp_usd) / cp_usd * 100
                        cv     = cp_usd * portfolio_holdings[ticker]["quantity"]
                        pv_    = cv * (1 + up / 100)
                        analyst_data.append({
                            "Ticker": ticker, "Currency": ccy,
                            f"Current Price ({ccy})": cp_native,
                            f"Target Price ({ccy})": tp_native,
                            "Current Price (USD)": cp_usd,
                            "Target Price (USD)": tp_usd,
                            "Upside/Downside": up,
                            "Analysts": info_.get("numberOfAnalystOpinions", 0),
                            "Recommendation": info_.get("recommendationKey", "N/A").upper(),
                            "Current Value (USD)": cv,
                            "Projected Value (USD)": pv_,
                            "Potential Gain (USD)": pv_ - cv,
                        })
                except Exception as e:
                    st.warning(f"Could not fetch analyst data for {ticker}: {e}")

            if analyst_data:
                da = pd.DataFrame(analyst_data)
                tc = da["Current Value (USD)"].sum()
                tp_ = da["Projected Value (USD)"].sum()
                tg  = da["Potential Gain (USD)"].sum()
                wu  = tg / tc * 100 if tc > 0 else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Portfolio Value (USD)",  f"${tc:,.0f}")
                c2.metric("Projected Value (12M, USD)",     f"${tp_:,.0f}", f"{wu:.1f}%")
                c3.metric("Potential Gain (USD)",           f"${tg:,.0f}")

                dd = da.copy()
                for col in ["Current Price (USD)","Target Price (USD)","Current Value (USD)","Projected Value (USD)","Potential Gain (USD)"]:
                    dd[col] = dd[col].apply(lambda x: f"${x:,.2f}")
                dd["Upside/Downside"] = dd["Upside/Downside"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(dd, use_container_width=True, hide_index=True)

                fig = go.Figure(go.Bar(
                    x=da["Ticker"], y=da["Upside/Downside"],
                    marker_color=["green" if x > 0 else "red" for x in da["Upside/Downside"]],
                    text=[f"{x:.1f}%" for x in da["Upside/Downside"]], textposition="outside",
                ))
                fig.update_layout(title="Analyst Consensus: Upside/Downside (USD-normalised)", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No analyst consensus data available.")

        # ── DCF ──────────────────────────────────────────────────────────────
        with forecast_method[2]:
            st.markdown("### **DCF Analysis**")
            st.info("Simplified DCF model — values remain in the stock's native currency.")
            sel = st.selectbox("Select Stock for DCF Analysis",
                               [t for t in portfolio_holdings if t in stock_data])
            if sel:
                ccy_sel = portfolio_holdings[sel]["currency"]
                c1, c2  = st.columns(2)
                with c1:
                    st.subheader(f"Input Parameters ({ccy_sel})")
                    try:
                        cf = yf.Ticker(sel).cashflow
                        fcf_default = int(abs(cf.loc["Free Cash Flow"].iloc[0])) if not cf.empty and "Free Cash Flow" in cf.index else 1_000_000_000
                    except Exception:
                        fcf_default = 1_000_000_000
                    fcf_in = st.number_input(f"Current Free Cash Flow ({ccy_sel})", 0, value=fcf_default, step=1_000_000)
                    gr     = st.slider("Growth Rate (%)", -10.0, 50.0, 5.0, 0.5)
                    py     = st.number_input("Projection (years)", 1, 10, 5)
                    wacc_  = st.slider("WACC (%)", 1.0, 20.0, 10.0, 0.5)
                    tg     = st.slider("Terminal Growth (%)", 0.0, 5.0, 2.5, 0.1)
                    try:    so = int(yf.Ticker(sel).info.get("sharesOutstanding", 1_000_000_000))
                    except: so = 1_000_000_000
                    shares = st.number_input("Shares Outstanding", 1_000_000, value=so, step=1_000_000)

                with c2:
                    st.subheader(f"DCF Calculation ({ccy_sel})")
                    pf = [fcf_in * (1 + gr / 100) ** y for y in range(1, py + 1)]
                    pv = [pf[i] / (1 + wacc_ / 100) ** (i + 1) for i in range(py)]
                    tv = pf[-1] * (1 + tg / 100) / (wacc_ / 100 - tg / 100)
                    pv_tv = tv / (1 + wacc_ / 100) ** py
                    ev   = sum(pv) + pv_tv
                    fv   = ev / shares
                    fv_usd = fv * _spot_fx(ccy_sel, "USD")
                    cp   = stock_data[sel]["current_price"]
                    cp_usd = cp * _spot_fx(ccy_sel, "USD")
                    ud   = (fv - cp) / cp * 100 if cp > 0 else 0

                    m1, m2 = st.columns(2)
                    m1.metric(f"Fair Value ({ccy_sel})", f"{fv:.2f}")
                    m1.metric("Fair Value (USD)",         f"${fv_usd:.2f}")
                    m1.metric("Enterprise Value",         f"{ccy_sel} {ev/1e9:.2f}B")
                    m2.metric(f"Current Price ({ccy_sel})", f"{cp:.2f}")
                    m2.metric("Current Price (USD)",         f"${cp_usd:.2f}")
                    m2.metric("Upside/Downside",             f"{ud:.1f}%", delta=f"{ud:.1f}%")

                    dcf_rows = [{"Year": y+1,
                                 f"Projected FCF ({ccy_sel})": f"{pf[y]/1e6:.1f}M",
                                 "Discount Factor": f"{1/(1+wacc_/100)**(y+1):.4f}",
                                 f"Present Value ({ccy_sel})": f"{pv[y]/1e6:.1f}M"} for y in range(py)]
                    st.dataframe(pd.DataFrame(dcf_rows), use_container_width=True, hide_index=True)

                    if ud > 20:    st.success(f"Undervalued by {ud:.1f}%")
                    elif ud < -20: st.error(f"Overvalued by {abs(ud):.1f}%")
                    else:          st.info("Fairly valued (within ±20%)")

# =============================================================================
# SECTOR PAGES
# =============================================================================
elif main_page in ["📱TMT Sector","🏦FIG Sector","🏭Industrials Sector",
                   "⚡PUI Sector","🛒Consumer Goods Sector","🏥Healthcare Sector"]:
                       
    sector_name = main_page.replace(" Sector", "")
    st.title(f"{sector_name} Sector Analysis")

    sector_holdings = {k: v for k, v in portfolio_holdings.items() if v["sector"] == sector_name}

    if not sector_holdings:
        st.warning(f"No holdings found in {sector_name} sector.")
    else:
        if "sector_tab" not in st.session_state:
            st.session_state.sector_tab = "Performance Analysis"

        st.markdown("### Select Analysis Type")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Performance Analysis", key=f"perf_{sector_name}", use_container_width=True):
                st.session_state.sector_tab = "Performance Analysis"; st.rerun()
        with c2:
            if st.button("Financial Analysis", key=f"fin_{sector_name}", use_container_width=True):
                st.session_state.sector_tab = "Financial Analysis"; st.rerun()
        with c3:
            if st.button("Company Specific", key=f"spec_{sector_name}", use_container_width=True):
                st.session_state.sector_tab = "Company Specific"; st.rerun()

        st.markdown("---")
        sector_tab = st.session_state.sector_tab

        # ── Performance Analysis ─────────────────────────────────────────────
        if sector_tab == "Performance Analysis":
            st.markdown(f"## **Performance Analysis** — {sector_name}")

            benchmarks = {"TMT":"XLK","FIG":"XLF","Industrials":"XLI",
                          "PUI":"XLB","Consumer Goods":"XLP","Healthcare":"XLV"}
            bm_ticker  = benchmarks.get(sector_name, "URTH")

            c1, c2 = st.columns(2)
            with c1: start_date = st.date_input("Start Date", pd.to_datetime("2025-11-06"), key=f"s_{sector_name}")
            with c2: end_date   = st.date_input("End Date",   pd.to_datetime("today"),       key=f"e_{sector_name}")

            if st.button("Generate Performance Analysis", type="primary", key=f"gen_{sector_name}"):
                if start_date >= end_date:
                    st.error("Start date must be before end date.")
                else:
                    try:
                        from scipy import stats as scipy_stats
                        with st.spinner(f"Fetching {sector_name} data…"):
                            bm_data = yf.download(bm_ticker, start=start_date, end=end_date, progress=False)
                            bm_close = bm_data["Close"].iloc[:,0] if isinstance(bm_data["Close"],pd.DataFrame) else bm_data["Close"]

                            used_ccy_s = {info["currency"] for info in sector_holdings.values()}
                            fx_usd_s   = {ccy: _fetch_fx_series(ccy, start_date, end_date, "USD") for ccy in used_ccy_s}

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

                        def get_price_s(ticker, date):
                            if ticker not in sector_data: return None
                            d  = sector_data[ticker]
                            cl = d["Close"].iloc[:,0] if isinstance(d["Close"],pd.DataFrame) else d["Close"]
                            av = cl.index[cl.index <= date]
                            return float(cl.loc[av[-1]]) if len(av) > 0 else None

                        def get_rate_s(currency, date):
                            return _fx_rate_on_date(currency, date, fx_usd_s.get(currency), "USD")

                        all_dates = bm_close.index
                        sector_first_trade = pd.Timestamp(
                            _tx[_tx["Ticker"].isin(sector_holdings.keys())]["Date"].min()
                        ).normalize()
                        active_sec_dates = all_dates[all_dates >= sector_first_trade]
                        sec_vals  = pd.Series(index=active_sec_dates, dtype=float)
                        for date in active_sec_dates:
                            dv = 0.0
                            for ticker, info in sector_holdings.items():
                                p = get_price_s(ticker, date)
                                if p is None: continue
                                dv += p * info["quantity"] * get_rate_s(info["currency"], date)
                            sec_vals[date] = dv

                        first_nonzero_s = sec_vals[sec_vals > 0].index.min()
                        sec_vals        = sec_vals.loc[first_nonzero_s:]

                        sec_ret = sec_vals.pct_change().dropna()
                        sec_ret = sec_ret[sec_ret.abs() <= 0.25]

                        bm_ret  = bm_close.pct_change().dropna()
                        common  = sec_ret.index.intersection(bm_ret.index)
                        sr = sec_ret.loc[common]; br = bm_ret.loc[common]

                        trs = ((sec_vals.iloc[-1]/sec_vals.iloc[0])-1)*100
                        trb = ((bm_close.iloc[-1]/bm_close.iloc[0])-1)*100

                        rf    = 0.02/252
                        exc   = sr - rf
                        sharpe = (exc.mean()/exc.std(ddof=1))*np.sqrt(252) if exc.std(ddof=1) != 0 else 0

                        rd = pd.DataFrame({"bm":br,"sec":sr}).dropna()
                        if len(rd) > 2:
                            sl, ic, r_val_s, *_ = scipy_stats.linregress(rd["bm"], rd["sec"])
                            beta = sl; alpha_a = ic * 252
                        else:
                            beta = alpha_a = 0.0

                        vol_s = sr.std(ddof=1)*np.sqrt(252)*100
                        vol_b = br.std(ddof=1)*np.sqrt(252)*100
                        cum   = (1+sr).cumprod()
                        mdd   = ((cum - cum.expanding().max())/cum.expanding().max()).min()*100
                        act   = sr - br; te = act.std(ddof=1)*np.sqrt(252)
                        ir    = (act.mean()*252)/te if te != 0 else 0

                        st.markdown("### **Key Performance Metrics** (USD-based)")
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
                        c1.metric("Alpha (Annualized)", f"{alpha_a*100:.2f}%")
                        c2.metric("Tracking Error",     f"{te*100:.2f}%")

                        st.markdown(f"### **{sector_name} vs {bm_ticker} Performance** (USD)")
                        sn = (sec_vals/sec_vals.iloc[0])*100
                        bn = (bm_close/bm_close.iloc[0])*100
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=sn.index, y=sn, name=f"{sector_name} Portfolio (USD)", mode="lines"))
                        fig.add_trace(go.Scatter(x=bn.index, y=bn, name=f"{bm_ticker} Benchmark",         mode="lines"))
                        fig.update_xaxes(rangeselector=dict(buttons=[
                            dict(count=1,label="1M",step="month",stepmode="backward"),
                            dict(count=6,label="6M",step="month",stepmode="backward"),
                            dict(count=1,label="1Y",step="year",stepmode="backward"),
                            dict(step="all",label="All"),
                        ]))
                        fig.update_layout(yaxis_title="Normalised Value (Base=100)", xaxis_title="Date")
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### **Holdings Performance Breakdown** (USD)")
                        hp = []
                        for ticker, info in sector_holdings.items():
                            if ticker in sector_data:
                                ip = init_prices[ticker]; cp_ = cur_prices_s[ticker]
                                ret_pct = ((cp_/ip)-1)*100
                                iv_usd = ip * get_rate_s(info["currency"], sec_vals.index[0])  * info["quantity"]
                                cv_usd = cp_ * get_rate_s(info["currency"], sec_vals.index[-1]) * info["quantity"]
                                wt     = iv_usd / sec_vals.iloc[0] * 100 if sec_vals.iloc[0] > 0 else 0
                                contrib = (cv_usd - iv_usd) / sec_vals.iloc[0] * 100 if sec_vals.iloc[0] > 0 else 0
                                hp.append({"Ticker":ticker,"Name":info["name"],
                                           "Ccy": info["currency"],
                                           "Return (%)":ret_pct,"Weight (%)":wt,"Contribution (%)":contrib})

                        pdf = pd.DataFrame(hp).sort_values("Return (%)", ascending=False)
                        st.dataframe(pdf.style.format({"Return (%)":"{:.2f}%","Weight (%)":"{:.2f}%","Contribution (%)":"{:.2f}%"})
                                     .background_gradient(subset=["Return (%)"], cmap="RdYlGn"),
                                     hide_index=True, use_container_width=True)

                        st.markdown("### **Risk Metrics**")
                        var95 = np.percentile(sr, 5) * 100
                        if len(sector_holdings) > 1:
                            rdict = {}
                            for ticker, info in sector_holdings.items():
                                if ticker in sector_data:
                                    d  = sector_data[ticker]
                                    cl = d["Close"].iloc[:,0] if isinstance(d["Close"],pd.DataFrame) else d["Close"]
                                    rdict[info["name"]] = cl.pct_change().dropna()
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
            st.markdown(f"## **Financial Analysis** — {sector_name}")

            if st.button("Generate Financial Analysis", type="primary", key=f"fin_gen_{sector_name}"):

                fin_rows  = []
                failed_tx = []
                status_ph = st.empty()
                tickers_list = list(sector_holdings.items())

                for i, (ticker, info) in enumerate(tickers_list):
                    status_ph.info(f"Fetching data for **{ticker}** ({i+1}/{len(tickers_list)})…")
                    ti, err = _yf_info_with_retry(ticker, max_attempts=4, base_delay=3.0)

                    if err:
                        status_ph.warning(f"Rate-limited on {ticker}, waiting 15 s then retrying…")
                        time.sleep(15)
                        ti, err2 = _yf_info_with_retry(ticker, max_attempts=3, base_delay=5.0)
                        if err2:
                            failed_tx.append((ticker, err2))
                            status_ph.warning(f"Skipping **{ticker}** after repeated failures.")
                            time.sleep(1)
                            continue

                    cf_df = bs_df = None
                    try:
                        tk    = yf.Ticker(ticker)
                        cf_df = tk.cash_flow
                        bs_df = tk.balance_sheet
                    except Exception:
                        pass

                    row = {"Ticker": ticker, "Name": info["name"], "Currency": info["currency"]}
                    row.update(_extract_ratios(ti, cf_df, bs_df))
                    fin_rows.append(row)
                    time.sleep(1.0)

                status_ph.empty()

                if failed_tx:
                    names = ", ".join(t for t, _ in failed_tx)
                    st.warning(
                        f"Could not retrieve data for **{names}** (rate-limited). "
                        "Their bars are omitted but the **industry benchmark still uses the full peer set**."
                    )

                if not fin_rows:
                    st.error("No financial data could be retrieved. Please wait a minute and try again.")
                    st.stop()

                fin_df = pd.DataFrame(fin_rows)
                st.success(f"Retrieved data for {len(fin_rows)}/{len(tickers_list)} holdings.")

                with st.spinner("Fetching sector industry benchmarks (cached for 1 h)…"):
                    ind_avgs = fetch_sector_industry_avg(sector_name)

                st.markdown("### **Key Financial Ratios**")
                ratio_cfg = [
                    ("P/E Ratio",    "Price-to-Earnings"),
                    ("P/B Ratio",    "Price-to-Book"),
                    ("Debt/Equity",  "Debt-to-Equity"),
                    ("OCF Ratio",    "Operating Cash Flow Ratio"),
                ]
                c1, c2 = st.columns(2)
                for idx, (col, title) in enumerate(ratio_cfg):
                    cd = fin_df[["Name", col]].dropna()
                    if cd.empty:
                        (c1 if idx % 2 == 0 else c2).caption(f"*{title}: no data available*")
                        continue
                    ind_avg  = ind_avgs.get(col)
                    port_avg = cd[col].mean()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=cd["Name"], y=cd[col], name=title,
                        marker_color="#030C30", text=cd[col].round(2), textposition="outside",
                    ))
                    if ind_avg is not None:
                        fig.add_trace(go.Scatter(
                            x=cd["Name"], y=[ind_avg]*len(cd), mode="lines",
                            name=f"Industry Median: {ind_avg:.2f}",
                            line=dict(color="#FF4B4B", width=2, dash="dot"),
                        ))
                    if len(cd) > 1:
                        fig.add_trace(go.Scatter(
                            x=cd["Name"], y=[port_avg]*len(cd), mode="lines",
                            name=f"Portfolio Avg: {port_avg:.2f}",
                            line=dict(color="#FFA500", width=1.5, dash="dashdot"),
                        ))
                    fig.update_layout(
                        title=title, height=400, showlegend=True,
                        hovermode="x unified", template="plotly_white",
                    )
                    (c1 if idx % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)

                st.markdown("### **Comprehensive Financial Summary**")
                st.dataframe(fin_df.fillna("N/A"), hide_index=True, use_container_width=True)

                st.markdown("### **Sector Statistics**")
                nd = fin_df.select_dtypes(include=[np.number])
                if not nd.empty:
                    st.dataframe(
                        nd.describe().T[["mean","50%","min","max","std"]]
                          .rename(columns={"50%":"median"}),
                        use_container_width=True,
                    )

                if ind_avgs:
                    st.markdown("### **Industry Benchmark Medians**")
                    bm_df = pd.DataFrame([{"Ratio": k, "Industry Median": round(v, 2)}
                                          for k, v in ind_avgs.items()])
                    st.dataframe(bm_df, hide_index=True, use_container_width=True)

        # ── Company Specific ─────────────────────────────────────────────────
        elif sector_tab == "Company Specific":
            st.markdown(f"## **Company Specific Analysis** — {sector_name}")

            comp_opts = {info["name"]: ticker for ticker, info in sector_holdings.items()}
            sel_name  = st.selectbox("Select Company", list(comp_opts.keys()), key=f"cs_{sector_name}")

            if sel_name:
                sel_ticker = comp_opts[sel_name]
                cinfo      = sector_holdings[sel_ticker]
                ccy        = cinfo["currency"]

                st.subheader(f"{sel_name} ({sel_ticker}) — Native currency: {ccy}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sector",        cinfo["sector"])
                c2.metric("Currency",      ccy)
                c3.metric("Quantity Held", f"{cinfo['quantity']:,}")
                c4.metric(f"Avg Buy Price ({ccy})", f"{cinfo['purchase_price']:.2f}")

                st.markdown("---")
                st.markdown("### **Transaction History** for This Stock")
                tx_stock = cinfo.get("_transactions", pd.DataFrame())
                if not tx_stock.empty:
                    tx_display = tx_stock.copy()
                    tx_display["Date"] = tx_display["Date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(tx_display[["Date","Ticker","Action","Quantity"]],
                                 use_container_width=True, hide_index=True)

                    tx_stock_sorted = tx_stock.sort_values("Date")
                    running = 0
                    dates_run, vals_run = [], []
                    for _, r in tx_stock_sorted.iterrows():
                        running += r["Quantity"] if r["Action"] == "Buy" else -r["Quantity"]
                        dates_run.append(r["Date"]); vals_run.append(running)
                    fig_tx = go.Figure(go.Scatter(x=dates_run, y=vals_run, mode="lines+markers",
                                                  line=dict(color="#0F1D64", width=2), marker=dict(size=8)))
                    fig_tx.update_layout(title="Running Net Position Over Time",
                                         xaxis_title="Date", yaxis_title="Net Shares Held",
                                         height=300, template="plotly_white")
                    st.plotly_chart(fig_tx, use_container_width=True)

                st.markdown("---")
                st.markdown(f"### **Stock Price Analysis** ({ccy})")
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

                if st.button("Generate Stock Analysis", type="primary", key=f"gsa_{sel_ticker}"):
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
                                                         name=f"{sel_ticker} Price ({ccy})",
                                                         line=dict(color="#0F1D64", width=2)))

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
                                    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                                                  y0=target_price, y1=target_price,
                                                  line=dict(color="red", width=2, dash="dash"))
                                    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                                             name=f"Target Price: {target_price:.2f} {ccy}",
                                                             line=dict(color="red", width=2, dash="dash"), showlegend=True))

                                fig.update_layout(
                                    title=f"{sel_name} – Stock Price History ({ccy})",
                                    xaxis_title="Date", yaxis_title=f"Price ({ccy})",
                                    hovermode="x unified", height=520, template="plotly_white",
                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                )
                                fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=[
                                    dict(count=1,label="1M",step="month",stepmode="backward"),
                                    dict(count=3,label="3M",step="month",stepmode="backward"),
                                    dict(count=6,label="6M",step="month",stepmode="backward"),
                                    dict(count=1,label="1Y",step="year",stepmode="backward"),
                                    dict(step="all",label="All"),
                                ]))
                                st.plotly_chart(fig, use_container_width=True)

                                cur_p    = float(cl.iloc[-1])
                                pp       = cinfo["purchase_price"]
                                tr_      = ((cur_p - pp) / pp * 100) if pp > 0 else 0
                                fx_spot  = _spot_fx(ccy, "USD")

                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric(f"Current Price ({ccy})",   f"{cur_p:.2f}",
                                          delta=f"{tr_:.2f}% vs avg buy")
                                c1.metric("Current Price (USD)",       f"${cur_p * fx_spot:.2f}")
                                c2.metric(f"Position Value ({ccy})",   f"{cur_p*cinfo['quantity']:,.2f}")
                                c2.metric("Position Value (USD)",       f"${cur_p*cinfo['quantity']*fx_spot:,.0f}")
                                c3.metric(f"Total Gain/Loss ({ccy})",   f"{(cur_p-pp)*cinfo['quantity']:,.2f}",
                                          delta=f"{tr_:.2f}%")
                                c3.metric("Total Gain/Loss (USD)",       f"${(cur_p-pp)*cinfo['quantity']*fx_spot:,.0f}")
                                if target_price and target_price > 0:
                                    c4.metric("Upside to Target", f"{((target_price-cur_p)/cur_p*100):.2f}%",
                                              delta=f"{target_price:.2f} {ccy} target")
                                else:
                                    c4.metric("Upside to Target", "N/A")

                                st.markdown("---")
                                st.markdown("### **Price Targets**")
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric(f"Your Target Price ({ccy})",   f"{target_price:.2f}" if target_price else "Not Set")
                                c2.metric(f"Analyst Consensus ({ccy})",   f"{an_target:.2f}" if an_target else "N/A")
                                c3.metric("Number of Analysts",           an_count if an_target else "N/A")
                                c4.metric("Recommendation",               an_rec)

                                st.markdown("---")
                                st.markdown("### **Investment Thesis**")
                                st.info(cinfo.get("thesis", "No thesis available"))

                                st.markdown("---")
                                st.markdown("### **DCF Valuation Parameters**")
                                c1, c2 = st.columns(2)
                                c1.write(f"**WACC:** {cinfo.get('WACC','N/A')}%")
                                with c2:
                                    st.write(f"**Cash Flow Projections ({ccy}):**")
                                    for i in range(1, 6):
                                        st.write(f"Year {i}: {cinfo.get(f'CF_{i}','N/A')}")

                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback; st.code(traceback.format_exc())

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.info("**Portfolio Dashboard v2.1**\n\nAll non-USD positions correctly converted using live FX rates.")
