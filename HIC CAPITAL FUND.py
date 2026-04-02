import json
import statistics as _stats
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
EXCEL_PATH    = "portfolio_holdings.xlsx"
CCY_CACHE     = Path("/Users/tevajanura/Downloads/currency_cache.json")
FUND_SIZE_USD = 1_000_000.0    #starting capital
RF_ANNUAL     = 0.02           # risk-free rate

# MSCI World sector ETFs used as benchmarks (yfinance tickers)
SECTOR_ETFS = {
    "TMT":            "WITS.AS",
    "FIG":            "WFNS.AS",
    "Industrials":    "WINS.AS",
    "PUI":            "WMTS.AS",
    "Consumer Goods": "WCOD.AS",
    "Healthcare":     "WHCS.AS",
    "Real Estate":    "WREI.AS",
    "Energy":         "WENS.L",
    "Utilities":      "WUTY.AS",
}
MSCI_WORLD = "URTH"   # broad MSCI World fallback benchmark

# Peer tickers for computing sector-median financial ratios (Financial Analysis tab).
# These are actual large-cap names from each iShares MSCI World sector fund.
# They are NOT the same as the ETF benchmark — the ETF is used for price comparison,
# these peers are used to pull ratio data (P/E, ROE, etc.) from yfinance.
SECTOR_PROXIES = {
    "WITS.AS": ["AAPL","MSFT","NVDA","AVGO","ORCL","SAP","2330.TW","ASML","CRM","ACN"],
    "WFNS.AS": ["JPM","BAC","GS","MS","BRK-B","HSBA.L","AXP","BLK","8306.T","RY"],
    "WINS.AS": ["GE","CAT","HON","RTX","UNP","SIE.DE","7011.T","ABB","ROK","ETN"],
    "WMTS.AS": ["LIN","APD","ECL","SHW","BHP","RIO","GLEN.L","NEM","FCX","ALB"],
    "WCOD.AS": ["PG","COST","WMT","KO","PEP","NESN.SW","ULVR.L","2914.T","OR.PA","PM"],
    "WHCS.AS": ["LLY","UNH","JNJ","MRK","ABBV","NVO","NOVN.SW","AZN","TMO","DHR"],
    "WREI.AS": ["AMT","PLD","EQIX","PSA","8951.T","GMG.AX","CCI","DLR","O","SPG"],
    "WENS.L":  ["XOM","CVX","SHEL","BP","TTE","COP","SLB","ENB","EOG","EQNR"],
    "WUTY.AS": ["NEE","DUK","SO","ENEL.MI","IBE.MC","SSE.L","EXC","AEP","RWE.DE","XEL"],
    "URTH":    ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","ASML","JPM","LLY"],
}

RATIO_COLS = (
    "P/E Ratio","P/B Ratio","Debt/Equity","OCF Ratio",
    "Forward P/E","Profit Margin (%)","ROE (%)","ROA (%)","Beta","Current Ratio",
)

# Fallback FX rates in case yfinance is unavailable
FALLBACK_USD = {
    "USD":1.0,"EUR":1.085,"GBP":1.27,"JPY":0.0067,"CHF":1.126,"CAD":0.735,
    "AUD":0.65,"HKD":0.128,"INR":0.012,"CNY":0.138,"SEK":0.094,"NOK":0.094,
    "DKK":0.144,"SGD":0.745,"KRW":0.00073,"BRL":0.19,"MXN":0.058,
}
FALLBACK_CHF = {
    "USD":0.888,"EUR":0.940,"GBP":1.120,"JPY":0.0059,"CHF":1.0,"CAD":0.650,
    "AUD":0.570,"HKD":0.114,"INR":0.0104,"CNY":0.122,"SEK":0.083,"NOK":0.083,
    "DKK":0.126,"SGD":0.660,"KRW":0.00065,
}

# Colours
NAVY = "#0F1D64"
RED  = "#FF6B6B"
BLUE_SCALE = ["#0F1D64","#1E3A8A","#3B82F6","#60A5FA","#93C5FD","#DBEAFE"]


# =============================================================================
# MODULE-LEVEL LOOKUP HELPERS
# These are plain functions (not closures) so they can cross @st.cache_data's
# pickle boundary.  They take the DataFrames as explicit arguments.
# =============================================================================

def price_on(ticker: str, date: pd.Timestamp, prices_df: pd.DataFrame) -> float | None:
    """Most recent closing price for `ticker` on or before `date`."""
    if ticker not in prices_df.columns:
        return None
    col   = prices_df[ticker].dropna()
    avail = col.index[col.index <= date]
    return float(col.loc[avail[-1]]) if len(avail) else None


def _get_fx_raw(ccy: str, date: pd.Timestamp, fx_df: pd.DataFrame, fallback: dict) -> float:
    """Look up FX rate from a pre-downloaded DataFrame; fall back to static table."""
    if ccy in fx_df.columns and not fx_df.empty:
        avail = fx_df.index[fx_df.index <= date]
        if len(avail):
            return float(fx_df.loc[avail[-1], ccy])
    return fallback.get(ccy, 1.0)


def get_fx_usd(ccy: str, date: pd.Timestamp, fx_usd_df: pd.DataFrame) -> float:
    """Return the rate: 1 unit of `ccy` → USD on `date`."""
    if ccy == "USD":
        return 1.0
    return _get_fx_raw(ccy, date, fx_usd_df, FALLBACK_USD)


def get_fx_chf(ccy: str, date: pd.Timestamp, fx_usd_df: pd.DataFrame, fx_chf_df: pd.DataFrame) -> float:
    """Return the rate: 1 unit of `ccy` → CHF on `date`."""
    if ccy == "CHF":
        return 1.0
    usd_rate = get_fx_usd(ccy, date, fx_usd_df)          # native → USD
    chf_rate = _get_fx_raw("USD", date, fx_chf_df, FALLBACK_CHF)  # USD → CHF
    return usd_rate * chf_rate


def _download_fx_pairs(pairs: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download a set of FX pairs and return a DataFrame[date × currency_code]."""
    tickers = [v for v in pairs.values() if v]
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        cl  = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        cl  = cl.ffill()
        reverse = {v: k for k, v in pairs.items() if v}
        cl.columns = [reverse.get(c, c) for c in cl.columns]
        return cl
    except Exception as e:
        print(f"FX download failed: {e}")
        return pd.DataFrame()


# =============================================================================
# CURRENCY CACHE
# =============================================================================
# Resolving a ticker's currency from yfinance is slow (~0.5 s per call) and
# burns rate-limit budget. We persist results to a JSON file on disk so
# subsequent restarts skip already-known currencies.

def _resolve_currencies(tickers: list[str]) -> dict[str, str]:
    """Return {ticker: ISO currency code}. Reads disk cache first."""
    disk: dict = {}
    if CCY_CACHE.exists():
        try:    disk = json.loads(CCY_CACHE.read_text())
        except: pass

    result, dirty = {}, False
    for t in tickers:
        cached = disk.get(t)
        if cached and isinstance(cached, str) and len(cached) == 3:
            result[t] = cached
            continue
        ccy = None
        for attempt in range(4):
            try:
                raw = getattr(yf.Ticker(t).fast_info, "currency", None) \
                      or yf.Ticker(t).info.get("currency")
                if raw and len(raw) == 3:
                    ccy = raw.upper(); break
            except Exception as e:
                time.sleep(2 ** attempt)
        result[t] = ccy or "USD"
        if ccy:
            disk[t] = ccy; dirty = True
        time.sleep(0.3)

    if dirty:
        try: CCY_CACHE.write_text(json.dumps(disk, indent=2))
        except: pass
    return result


# =============================================================================
# EXCEL LOADING
# =============================================================================

def _read_excel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Excel file and return (transactions_df, info_df).

    transactions_df has columns: Date, Ticker, Action, Quantity, YF_Ticker
    info_df         has one row per ticker: name, sector, currency, target_price,
                    thesis, WACC, CF_1..CF_5
    """
    xl = pd.ExcelFile(EXCEL_PATH)

    # Auto-detect the sheet that contains Buy / Sell rows
    raw = None
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns:
            vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
            if any(v in ["buy","sell"] for v in vals):
                raw = df; break
        if raw is not None: break

    if raw is None:
        raise ValueError("No sheet with Buy/Sell data found.")

    col_lower = {c.lower(): c for c in raw.columns}

    def find_col(candidates, fallback_idx):
        for k in candidates:
            if k in col_lower: return col_lower[k]
        return raw.columns[fallback_idx]

    date_col   = find_col(["date","trade date","tradedate"], 0)
    ticker_col = find_col(["ticker","security","symbol","stock"], 1)
    action_col = find_col(["action","type","side","buy/sell"], 2)
    qty_col    = find_col(["shares","quantity","qty","units","volume"], 3)

    tx = raw[[date_col, ticker_col, action_col, qty_col]].copy()
    tx.columns = ["Date","Ticker","Action","Quantity"]
    tx = tx.dropna(subset=["Date","Ticker","Action","Quantity"])
    tx["Date"]     = pd.to_datetime(tx["Date"], errors="coerce")
    tx             = tx.dropna(subset=["Date"])
    tx["Ticker"]   = tx["Ticker"].astype(str).str.strip()
    tx["Action"]   = tx["Action"].astype(str).str.strip().str.capitalize()
    tx["Quantity"] = pd.to_numeric(
        tx["Quantity"].astype(str).str.replace(",","").str.replace(" ",""),
        errors="coerce"
    ).fillna(0).astype(int)
    tx = tx[tx["Quantity"] > 0]
    tx["YF_Ticker"] = tx["Ticker"]

    # --- Extra metadata columns (sector, thesis, target price, etc.) ---
    core = {date_col, ticker_col, action_col, qty_col}
    extra_lower = {c.lower(): c for c in raw.columns if c not in core}

    FIELDS = {
        "name":         ["name","company","company name"],
        "target_price": ["target_price","target price","targetprice","target","tp",
                         "price target","pt","target px","targetpx"],
        "sector":       ["sector","industry","gics sector"],
        "thesis":       ["thesis","investment thesis","rationale"],
        "wacc":         ["wacc"],
        "cf_1":         ["cf_1","cf1","cashflow_1","cashflow1"],
        "cf_2":         ["cf_2","cf2","cashflow_2","cashflow2"],
        "cf_3":         ["cf_3","cf3","cashflow_3","cashflow3"],
        "cf_4":         ["cf_4","cf4","cashflow_4","cashflow4"],
        "cf_5":         ["cf_5","cf5","cashflow_5","cashflow5"],
    }

    def resolve(field):
        for k in FIELDS[field]:
            if k in extra_lower: return extra_lower[k]
        return None

    unique_tickers = tx["Ticker"].unique()
    currencies     = _resolve_currencies(list(unique_tickers))

    records = {}
    for ticker in unique_tickers:
        rows = raw[raw[ticker_col].astype(str).str.strip() == ticker]
        rec  = {"currency": currencies[ticker]}
        for field, defaults in [
            ("name", ticker), ("target_price", 0.0), ("sector", "Unknown"),
            ("thesis", ""), ("wacc", ""), ("cf_1",""), ("cf_2",""),
            ("cf_3",""), ("cf_4",""), ("cf_5",""),
        ]:
            col = resolve(field)
            if col:
                vals = rows[col].dropna()
                vals = vals[vals.astype(str).str.strip() != ""]
                rec[field] = vals.iloc[0] if len(vals) else defaults
            else:
                rec[field] = defaults
        records[ticker] = rec

    info_df = pd.DataFrame.from_dict(records, orient="index")
    info_df.index.name = "Ticker"
    return tx, info_df


# =============================================================================
# MASTER MARKET DATA  — fetched ONCE, cached by Streamlit
# =============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data() -> dict:
    """
    Single entry point for ALL market data.

    Downloads:
      1. transactions + info from Excel
      2. price history for every held ticker + every sector ETF + URTH
      3. FX history for every currency present in the portfolio → USD and → CHF
      4. Derives:
           - portfolio  : {ticker: {quantity, name, sector, currency, purchase_price, …}}
           - prices_df  : DataFrame[date × ticker], native-currency closing prices
           - fx_usd_df  : DataFrame[date × currency], rate currency→USD each day
           - nav_df     : DataFrame[date], columns: equity_usd, cash_usd, nav_usd
           - spot_usd   : {currency: float}  current spot rate → USD
           - spot_chf   : {currency: float}  current spot rate → CHF

    Returns a dict so callers can do:
        md = load_market_data()
        md["prices_df"], md["nav_df"], etc.
    """

    # ── 1. Read Excel ──────────────────────────────────────────────────────────
    tx, info_df = _read_excel()

    # ── 2. Build portfolio dict (net positions, VWAP cost, metadata) ──────────
    #    We need this before downloading prices so we know which dates matter.
    portfolio = {}
    for yf_ticker, grp in tx.groupby("YF_Ticker"):
        if pd.isna(yf_ticker): continue
        buys  = grp[grp["Action"] == "Buy"]
        sells = grp[grp["Action"] == "Sell"]
        net   = int(buys["Quantity"].sum()) - int(sells["Quantity"].sum())
        if net <= 0: continue

        row = info_df.loc[yf_ticker] if yf_ticker in info_df.index else pd.Series()

        def sf(v, d=0.0):
            try:    return float(v)
            except: return d
        def ss(v, d=""):
            return str(v) if pd.notna(v) else d

        portfolio[yf_ticker] = {
            "quantity":      net,
            "name":          ss(row.get("name", yf_ticker), yf_ticker),
            "Target_price":  sf(row.get("target_price", 0.0)),
            "currency":      ss(row.get("currency","USD"), "USD"),
            "sector":        ss(row.get("sector","Unknown"), "Unknown"),
            "purchase_date": buys["Date"].min().strftime("%Y-%m-%d"),
            "purchase_price": 0.0,   # filled below after prices are downloaded
            "thesis":        ss(row.get("thesis","")),
            "WACC":          ss(row.get("wacc","")),
            "CF_1":          ss(row.get("cf_1","")),
            "CF_2":          ss(row.get("cf_2","")),
            "CF_3":          ss(row.get("cf_3","")),
            "CF_4":          ss(row.get("cf_4","")),
            "CF_5":          ss(row.get("cf_5","")),
            "_transactions": grp.reset_index(drop=True),
        }

    # ── 3. Determine date range ───────────────────────────────────────────────
    first_trade = tx["Date"].min()
    today       = pd.Timestamp.today().normalize()

    # ── 4. Download ALL prices in one batch ───────────────────────────────────
    #    tickers = holdings + all sector ETFs + broad MSCI World benchmark
    all_tickers = (
        list(portfolio.keys())
        + list(SECTOR_ETFS.values())
        + [MSCI_WORLD]
    )
    all_tickers = list(dict.fromkeys(all_tickers))   # deduplicate, preserve order

    print(f"Downloading prices for {len(all_tickers)} tickers from {first_trade.date()} …")
    raw_prices = yf.download(
        all_tickers,
        start=first_trade - pd.Timedelta(days=10),   # 10 extra days for VWAP lookback
        end=today + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=True,
    )

    # yf.download returns MultiIndex columns (field, ticker) when >1 ticker
    close = raw_prices["Close"] if isinstance(raw_prices.columns, pd.MultiIndex) else raw_prices[["Close"]]
    # close is now a DataFrame: index=date, columns=ticker
    close = close.ffill()   # forward-fill weekends / holidays

    prices_df = close   # one column per ticker, native currency prices

    # ── 5. Download FX rates ──────────────────────────────────────────────────
    # Exclude USD (always 1.0) and CHF (CHF→CHF = 1.0; USD→CHF handled separately).
    currencies_needed = list({info["currency"] for info in portfolio.values()} - {"USD", "CHF"})

    fx_pairs_usd = {ccy: f"{ccy}USD=X" for ccy in currencies_needed}
    # For CHF we only need the USD→CHF rate; other currencies go via native→USD→CHF.
    fx_pairs_chf = {"USD": "USDCHF=X"}

    dl_start   = first_trade - pd.Timedelta(days=10)
    dl_end     = today + pd.Timedelta(days=1)
    fx_usd_raw = _download_fx_pairs(fx_pairs_usd, dl_start, dl_end)
    fx_chf_raw = _download_fx_pairs(fx_pairs_chf, dl_start, dl_end)

    # ── 6. Compute VWAP purchase prices now that prices_df is available ───────
    # Module-level price_on / get_fx_usd are used with explicit df arguments.
    for ticker, info in portfolio.items():
        buys     = info["_transactions"][info["_transactions"]["Action"] == "Buy"]
        total_cost, total_qty = 0.0, 0
        for _, row in buys.iterrows():
            p = price_on(ticker, row["Date"], prices_df) or 0.0
            q = int(row["Quantity"])
            total_cost += p * q
            total_qty  += q
        info["purchase_price"] = total_cost / total_qty if total_qty else 0.0

    # ── 7. Compute NAV time series ─────────────────────────────────────────────
    #    Walk every trading day in prices_df, apply transactions in order,
    #    and record: cash, equity value (USD), total NAV (USD).
    all_dates = prices_df.index

    # Group transactions by normalized date for O(1) lookup
    tx_by_date: dict[pd.Timestamp, list] = {}
    for _, row in tx.sort_values("Date").iterrows():
        d = row["Date"].normalize()
        tx_by_date.setdefault(d, []).append(row)

    cash_usd    = FUND_SIZE_USD
    live_shares = {t: 0 for t in tx["Ticker"].unique()}

    nav_rows = []
    for date in all_dates:
        date_norm = pd.Timestamp(date).normalize()

        # Apply trades on this date
        for tr in tx_by_date.get(date_norm, []):
            ticker = tr["Ticker"]
            qty    = int(tr["Quantity"])
            p_nat  = price_on(ticker, date, prices_df) \
                     or portfolio.get(ticker, {}).get("purchase_price", 0.0)
            p_usd  = p_nat * get_fx_usd(portfolio.get(ticker, {}).get("currency","USD"), date, fx_usd_raw)
            if tr["Action"] == "Buy":
                cash_usd -= p_usd * qty
                live_shares[ticker] = live_shares.get(ticker, 0) + qty
            elif tr["Action"] == "Sell":
                cash_usd += p_usd * qty
                live_shares[ticker] = max(0, live_shares.get(ticker, 0) - qty)

        # Mark equity to market in USD
        equity_usd = sum(
            price_on(t, date, prices_df) * shares * get_fx_usd(portfolio.get(t, {}).get("currency","USD"), date, fx_usd_raw)
            for t, shares in live_shares.items()
            if shares > 0 and price_on(t, date, prices_df) is not None
        )
        nav_rows.append({"date": date, "equity_usd": equity_usd,
                         "cash_usd": cash_usd, "nav_usd": equity_usd + cash_usd,
                         "live_shares": dict(live_shares)})   # snapshot for treemap etc.

    nav_df = pd.DataFrame(nav_rows).set_index("date")

    # ── 8. Current spot FX rates ───────────────────────
    spot_usd = {"USD": 1.0}
    spot_chf = {"CHF": 1.0}
    for ccy in currencies_needed:
        spot_usd[ccy] = get_fx_usd(ccy, today, fx_usd_raw)
        spot_chf[ccy] = get_fx_chf(ccy, today, fx_usd_raw, fx_chf_raw)
    # USD→CHF spot
    spot_chf["USD"] = _get_fx_raw("USD", today, fx_chf_raw, FALLBACK_CHF)

    return {
        "tx":         tx,
        "info_df":    info_df,
        "portfolio":  portfolio,
        "prices_df":  prices_df,
        "fx_usd_df":  fx_usd_raw,
        "fx_chf_df":  fx_chf_raw,
        "nav_df":     nav_df,
        "spot_usd":   spot_usd,
        "spot_chf":   spot_chf,
    }


# =============================================================================
# FINANCIAL RATIO HELPERS  (yfinance .info calls — not price data)
# =============================================================================

def _yf_info_retry(ticker: str, max_attempts: int = 4, base_delay: float = 3.0):
    CORE = ("marketCap","trailingPE","priceToBook","returnOnEquity","currentPrice")
    for attempt in range(max_attempts):
        try:
            info = yf.Ticker(ticker).info
            if info and any(info.get(f) is not None for f in CORE):
                return info, None
            raise ValueError("Rate-limit stub")
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(base_delay * 2 ** attempt)
            else:
                return {}, str(e)
    return {}, "Max retries"


def _extract_ratios(ti: dict, cf_df=None, bs_df=None) -> dict:
    ocf_r = None
    if cf_df is not None and bs_df is not None:
        try:
            ocf = cf_df.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf_df.index else None
            cl_ = bs_df.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in bs_df.index else None
            ocf_r = float(ocf)/float(cl_) if ocf and cl_ and float(cl_) != 0 else None
        except: pass
    return {
        "Market Cap":         ti.get("marketCap"),
        "P/E Ratio":          ti.get("trailingPE"),
        "Forward P/E":        ti.get("forwardPE"),
        "P/B Ratio":          ti.get("priceToBook"),
        "Dividend Yield (%)": (ti.get("dividendYield") or 0)*100,
        "Profit Margin (%)":  (ti.get("profitMargins")  or 0)*100,
        "ROE (%)":            (ti.get("returnOnEquity") or 0)*100,
        "ROA (%)":            (ti.get("returnOnAssets") or 0)*100,
        "Debt/Equity":        ti.get("debtToEquity"),
        "Current Ratio":      ti.get("currentRatio"),
        "Revenue Growth (%)": (ti.get("revenueGrowth")  or 0)*100,
        "Beta":               ti.get("beta"),
        "OCF Ratio":          ocf_r,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sector_avg(sector: str) -> dict:          # ← keep original name
    etf_ticker = SECTOR_ETFS.get(sector, MSCI_WORLD)
    proxies    = SECTOR_PROXIES.get(etf_ticker, SECTOR_PROXIES[MSCI_WORLD])
    peer_vals  = {col: [] for col in RATIO_COLS}
    for pt in proxies:
        info, err = _yf_info_retry(pt, max_attempts=3, base_delay=2.0)
        if err:
            continue
        mapping = {
            "P/E Ratio":         info.get("trailingPE"),
            "P/B Ratio":         info.get("priceToBook"),
            "Debt/Equity":       info.get("debtToEquity"),
            "OCF Ratio":         None,
            "Forward P/E":       info.get("forwardPE"),
            "Profit Margin (%)": (info.get("profitMargins")  or 0) * 100,
            "ROE (%)":           (info.get("returnOnEquity") or 0) * 100,
            "ROA (%)":           (info.get("returnOnAssets") or 0) * 100,
            "Beta":              info.get("beta"),
            "Current Ratio":     info.get("currentRatio"),
        }
        for col in RATIO_COLS:
            v = mapping.get(col)
            if v is None:
                continue
            try:
                fv = float(v)
                if col in ("P/E Ratio", "Forward P/E") and (fv < 0 or fv > 200):
                    continue
                if col == "Debt/Equity" and fv > 500:
                    continue
                peer_vals[col].append(fv)
            except Exception:
                pass
        time.sleep(0.5)
    return {col: _stats.median(vals) for col, vals in peer_vals.items() if vals}

# =============================================================================
# PAGE CONFIG & CSS
# =============================================================================
st.set_page_config(page_title="HIC Capital Dashboard", page_icon="", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"] { background-color: #030C30; }
[data-testid="stSidebar"] * { color: white !important; }
.main { background-color: white; }
div.stButton > button {
    height:120px; white-space:pre-wrap;
    background-color:#f0f2f6; border:2px solid #e0e2e6;
    color:#0F1D64; font-size:16px; font-weight:500; transition:all 0.3s ease;
}
div.stButton > button:hover {
    background-color:#0F1D64; color:white; border-color:#0F1D64;
    transform:translateY(-2px); box-shadow:0 4px 12px rgba(15,29,100,0.3);
}
[data-testid="stSidebar"] div.stButton > button {
    height:auto; background-color:rgba(255,255,255,0.1);
    border:1px solid rgba(255,255,255,0.3); color:white;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background-color:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.5);
    transform:none; box-shadow:none;
}
</style>""", unsafe_allow_html=True)

# =============================================================================
# LOAD ALL DATA  — runs once; Streamlit caches the result
# =============================================================================
with st.spinner("Loading market data (runs once, then cached)…"):
    md = load_market_data()

# Unpack for convenience throughout the file
tx         = md["tx"]
info_df    = md["info_df"]
portfolio  = md["portfolio"]
prices_df  = md["prices_df"]   # DataFrame[date × ticker], native prices
fx_usd_df  = md["fx_usd_df"]
fx_chf_df  = md["fx_chf_df"]
nav_df     = md["nav_df"]      # DataFrame[date] → equity_usd, cash_usd, nav_usd
spot_usd   = md["spot_usd"]    # {ccy: float}
spot_chf   = md["spot_chf"]
# price_on(ticker, date, prices_df) and get_fx_usd/chf(ccy, date, fx_df)
# are module-level functions — just use them directly with the unpacked dfs above.


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("Navigation")
try:
    st.sidebar.image("HIC_Capital_Logo.png", width='stretch')
    st.sidebar.markdown("---")
except Exception:
    st.sidebar.markdown("### Portfolio Dashboard")
    st.sidebar.markdown("---")

if "main_page" not in st.session_state:
    st.session_state.main_page = "Home"

if st.sidebar.button("🏠 Home",           width='stretch'): st.session_state.main_page = "Home"
if st.sidebar.button("📋 Transactions",   width='stretch'): st.session_state.main_page = "Transactions"
st.sidebar.markdown("---")
st.sidebar.subheader("Sectors")
for label, key in [("📱 TMT","TMT"),("🏦 FIG","FIG"),("🏭 Industrials","Industrials"),
                    ("⚡ PUI","PUI"),("🛒 Consumer Goods","Consumer Goods"),("🏥 Healthcare","Healthcare"),("🏢 Real Estate","Real Estate")]:
    if st.sidebar.button(label, width='stretch'): st.session_state.main_page = key

main_page = st.session_state.main_page


# =============================================================================
# SHARED HELPERS  (used by multiple pages, rely on prices_df / nav_df)
# =============================================================================

def price_series(ticker: str) -> pd.Series:
    """Closing price series for a single ticker (native currency)."""
    if ticker in prices_df.columns:
        return prices_df[ticker].dropna()
    return pd.Series(dtype=float)


def sector_prices_usd(sector_holdings: dict, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns a DataFrame[date × ticker] where each cell is the price in USD.
    Used by both Performance Analysis and the NAV computation.
    """
    result = pd.DataFrame(index=date_range, columns=list(sector_holdings.keys()), dtype=float)
    for ticker, info in sector_holdings.items():
        if ticker not in prices_df.columns:
            continue
        native = prices_df[ticker].reindex(date_range, method="ffill")
        for date in date_range:
            p = native.loc[date] if date in native.index else None
            if p and not np.isnan(p):
                result.loc[date, ticker] = p * get_fx_usd(info["currency"], date, fx_usd_df)
    return result.astype(float)


def current_price(ticker: str) -> float | None:
    """Latest available closing price for a ticker (native currency)."""
    s = price_series(ticker)
    return float(s.iloc[-1]) if not s.empty else None


# =============================================================================
# TRANSACTION HISTORY PAGE
# =============================================================================
if main_page == "Transactions":
    st.title("Transaction History")

    st.markdown("**All Transactions**")
    display_tx = tx.reset_index(drop=True).copy()

    # Execution price = closing price on trade date (already in prices_df)
    display_tx["Exec Price"] = display_tx.apply(
        lambda r: price_on(r["Ticker"], r["Date"], prices_df), axis=1
    )
    display_tx["Currency"]   = display_tx["Ticker"].apply(
        lambda t: portfolio.get(t, {}).get("currency", "")
    )
    display_tx["Gross Value"] = display_tx.apply(
        lambda r: r["Exec Price"] * r["Quantity"] if pd.notna(r["Exec Price"]) else None, axis=1
    )
    display_tx["Date"]        = display_tx["Date"].dt.strftime("%Y-%m-%d")
    display_tx["Exec Price"]  = display_tx["Exec Price"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    display_tx["Gross Value"] = display_tx["Gross Value"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    st.dataframe(display_tx[["Date","Ticker","Action","Quantity","Currency","Exec Price","Gross Value"]],
                 width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("**Net Position Summary**")
    rows = []
    for yf_ticker, grp in tx.groupby("YF_Ticker"):
        if pd.isna(yf_ticker): continue
        bought = int(grp[grp["Action"]=="Buy"]["Quantity"].sum())
        sold   = int(grp[grp["Action"]=="Sell"]["Quantity"].sum())
        net    = bought - sold
        rows.append({"YF Ticker": yf_ticker, "Bought": bought, "Sold": sold,
                     "Net Position": net, "Status": "🟢 Open" if net > 0 else "🔴 Closed"})
    st.dataframe(pd.DataFrame(rows).sort_values("YF Ticker"), width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("**Transaction Timeline** (bubble size = trade value in USD)")
    bubble_tx = tx.reset_index(drop=True).copy()
    bubble_tx["exec_price_usd"] = bubble_tx.apply(
        lambda r: (price_on(r["Ticker"], r["Date"], prices_df) or 0)
                  * spot_usd.get(portfolio.get(r["Ticker"],{}).get("currency","USD"), 1.0),
        axis=1
    )
    bubble_tx["gross_value"] = bubble_tx["exec_price_usd"] * bubble_tx["Quantity"]
    max_val = bubble_tx["gross_value"].max()
    bubble_tx["bubble_size"] = (bubble_tx["gross_value"] / max_val) * 42 + 8

    fig = go.Figure()
    for action, color in [("Buy", NAVY), ("Sell", RED)]:
        sub = bubble_tx[bubble_tx["Action"] == action]
        fig.add_trace(go.Scatter(
            x=sub["Date"], y=sub["Ticker"], mode="markers",
            marker=dict(color=color, size=sub["bubble_size"], opacity=0.8, sizemode="diameter"),
            name=action,
            hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Qty: %{customdata[0]:,}<br>Value: $%{customdata[1]:,.0f}<extra></extra>",
            customdata=sub[["Quantity","gross_value"]].values,
        ))
    fig.update_layout(title="Buy & Sell Events", xaxis_title="Date", yaxis_title="Ticker",
                      height=520, template="plotly_white")
    st.plotly_chart(fig, width='stretch')


# =============================================================================
# HOME PAGE
# =============================================================================
elif main_page == "Home":
    st.title("Portfolio Dashboard - Home")

    if "home_tab" not in st.session_state:
        st.session_state.home_tab = "Generic Summary"

    st.markdown("### **Select Analysis Type**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Generic Summary\n\nKey metrics, performance vs MSCI World, portfolio treemap",
                     key="btn_gen", width='stretch'):
            st.session_state.home_tab = "Generic Summary"
    with c2:
        if st.button("Portfolio Structure\n\nSector, geographical, and asset distribution",
                     key="btn_str", width='stretch'):
            st.session_state.home_tab = "Portfolio Structure Analysis"
    with c3:
        if st.button("Forecast (still under dev.)\n\nMonte Carlo, analyst targets, DCF analysis",
                     key="btn_frc", width='stretch'):
            st.session_state.home_tab = "Forecast"
    st.markdown("---")
    home_tab = st.session_state.home_tab

    # =========================================================================
    # HOME — Generic Summary
    # =========================================================================
    if home_tab == "Generic Summary":
        st.markdown("## **Generic Summary**")

        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input("Start Date", pd.to_datetime("2025-11-06").date())
        with c2: end_date   = st.date_input("End Date",   pd.Timestamp.today().date())

        if st.button("Generate Analysis", type="primary"):
            if start_date >= end_date:
                st.error("Start date must be before end date."); st.stop()

            # Slice the pre-computed NAV to the requested window
            nav_slice  = nav_df.loc[str(start_date):str(end_date)]
            msci_slice = price_series(MSCI_WORLD).loc[str(start_date):str(end_date)]

            if nav_slice.empty or msci_slice.empty:
                st.error("No data in that date range."); st.stop()

            # ── Risk Metrics ──────────────────────────────────────────────────
            st.markdown("### **Portfolio Risk Metrics**")
            first_trade_in_range = tx["Date"].min().normalize()
            active_nav = nav_slice[nav_slice.index >= first_trade_in_range]["nav_usd"]
            active_nav = active_nav[active_nav > 0]
            port_ret   = active_nav.pct_change().dropna()
            port_ret   = port_ret[port_ret.abs() <= 0.25]  # filter data glitches

            msci_ret  = msci_slice.pct_change().dropna()
            common    = port_ret.index.intersection(msci_ret.index)

            if len(common) >= 10:
                pr, mr  = port_ret.loc[common], msci_ret.loc[common]
                rf_d    = RF_ANNUAL / 252
                exc     = pr - rf_d
                sharpe  = (exc.mean() / exc.std(ddof=1)) * np.sqrt(252) if exc.std(ddof=1) else 0

                from scipy import stats as scipy_stats
                reg = pd.DataFrame({"m": mr, "p": pr}).dropna()
                if len(reg) > 2:
                    slope, intercept, r_val, *_ = scipy_stats.linregress(reg["m"], reg["p"])
                    beta, alpha_ann, r_sq = slope, intercept * 252, r_val**2
                else:
                    beta = alpha_ann = r_sq = 0.0

                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe Ratio",         f"{sharpe:.2f}")
                c2.metric("Beta (vs MSCI World)", f"{beta:.2f}")
                c3.metric("Alpha (Ann.)",          f"{alpha_ann*100:.2f}%")
                c1, c2, c3 = st.columns(3)
                c1.metric("Portfolio Vol (Ann.)",  f"{pr.std(ddof=1)*np.sqrt(252)*100:.1f}%")
                c2.metric("MSCI World Vol (Ann.)", f"{mr.std(ddof=1)*np.sqrt(252)*100:.1f}%")
                c3.metric("R² (vs MSCI World)",   f"{r_sq:.2f}")

                def sharpe_label(s):
                    return "Good" if s>1.5 else "Acceptable" if s>0.5 else "Poor" if s>0 else "Negative"
                st.info(f"""
**Interpretation ({len(pr)} trading days):**
- Sharpe {sharpe:.2f} → {sharpe_label(sharpe)}
- Beta {beta:.2f} → {'more volatile' if beta>1 else 'less volatile'} than MSCI World
- Alpha {alpha_ann*100:.2f}% p.a. → {'outperforming' if alpha_ann>0 else 'underperforming'}
- R² {r_sq:.2f} → {r_sq*100:.0f}% of variance explained by MSCI World
""")
            else:
                st.warning("Not enough overlapping trading days to compute risk metrics.")

            # ── NAV vs MSCI World chart ────────────────────────────────────────
            st.markdown("### **Portfolio vs MSCI World**")
            nav_norm  = nav_slice["nav_usd"] / FUND_SIZE_USD * 100
            msci_norm = msci_slice / msci_slice.iloc[0] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav_norm.index, y=nav_norm, name="Fund NAV (USD)",
                                     line=dict(color=NAVY, width=3),
                                     hovertemplate="<b>NAV</b> %{x}<br>$%{customdata:,.0f}<extra></extra>",
                                     customdata=nav_slice["nav_usd"].values))
            fig.add_trace(go.Scatter(x=msci_norm.index, y=msci_norm, name="MSCI World (URTH)",
                                     line=dict(color=RED, width=2)))
            fig.update_layout(title="Fund NAV vs MSCI World (Base 100 = $1,000,000)",
                              xaxis_title="Date", yaxis_title="Index (Base 100)",
                              hovermode="x unified", height=520, template="plotly_white")
            st.plotly_chart(fig, width='stretch')

            # KPI metrics
            final = nav_slice.iloc[-1]
            final_nav    = final["nav_usd"]
            final_equity = final["equity_usd"]
            final_cash   = final["cash_usd"]
            port_ret_tot = (final_nav - FUND_SIZE_USD) / FUND_SIZE_USD * 100
            msci_ret_tot = (msci_slice.iloc[-1] - msci_slice.iloc[0]) / msci_slice.iloc[0] * 100
            outperf      = port_ret_tot - msci_ret_tot
            cash_pct     = final_cash / final_nav * 100 if final_nav else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fund NAV (USD)",     f"${final_nav:,.0f}",    delta=f"{port_ret_tot:.2f}%")
            c2.metric("Equity Value (USD)", f"${final_equity:,.0f}", delta=f"{final_equity/final_nav*100:.1f}% of NAV")
            c3.metric("Cash (USD)",          f"${final_cash:,.0f}",  delta=f"{cash_pct:.1f}% of NAV")
            c4.metric("vs MSCI World",       f"{outperf:+.2f}%")

            # ── Treemap ────────────────────────────────────────────────────────
            st.markdown("### **Portfolio Composition Treemap** (USD)")
            final_date  = nav_slice.index[-1]
            live_shares = nav_slice.loc[final_date, "live_shares"]

            treemap_rows = []
            for ticker, shares in live_shares.items():
                if shares <= 0: continue
                info  = portfolio.get(ticker)
                if info is None: continue
                p_nat = price_on(ticker, final_date, prices_df)
                if p_nat is None: continue
                ccy     = info["currency"]
                p_usd   = p_nat * get_fx_usd(ccy, final_date, fx_usd_df)
                val_usd = p_usd * shares

                # Average cost in USD
                buy_tx   = tx[(tx["Ticker"]==ticker) & (tx["Action"]=="Buy")]
                cost_usd = sum(
                    (price_on(ticker, r["Date"], prices_df) or info["purchase_price"])
                    * get_fx_usd(ccy, r["Date"], fx_usd_df) * int(r["Quantity"])
                    for _, r in buy_tx.iterrows()
                )
                avg_cost = cost_usd / shares if shares else 0
                perf     = (p_usd - avg_cost) / avg_cost * 100 if avg_cost else 0

                treemap_rows.append({"ticker": ticker, "name": info["name"],
                                     "weight": val_usd / final_nav * 100,
                                     "performance": perf, "value": val_usd,
                                     "currency": ccy})

            treemap_rows.append({"ticker":"CASH","name":"Cash (USD)",
                                 "weight": final_cash/final_nav*100,
                                 "performance":0.0,"value":final_cash,"currency":"USD"})
            tm_df = pd.DataFrame(treemap_rows)

            fig_tm = px.treemap(
                tm_df, path=[px.Constant("Portfolio"),"name"],
                values="weight", color="performance",
                color_continuous_scale=[[0, RED],[0.5,"#FFFFFF"],[1, NAVY]],
                color_continuous_midpoint=0,
                hover_data={"weight":":.2f","performance":":.2f","value":":,.0f","currency":True},
                labels={"weight":"Weight (%)","performance":"Return (%)","value":"Value (USD)"},
            )
            fig_tm.update_traces(
                textposition="middle center",
                texttemplate="<b>%{label}</b><br>%{value:.1f}%",
                hovertemplate="<b>%{label}</b><br>Weight: %{customdata[0]:.2f}%<br>Return: %{customdata[1]:.2f}%<br>Value: $%{customdata[2]:,.0f}<br>Ccy: %{customdata[3]}<extra></extra>",
            )
            fig_tm.update_layout(height=600, margin=dict(t=50, l=0, r=0, b=0))
            st.plotly_chart(fig_tm, use_container_width=True)

            # Holdings table
            ht = tm_df.copy()
            ht["Return (%)"]  = ht["performance"].apply(lambda x: f"{x:.2f}%")
            ht["Weight (%)"]  = ht["weight"].apply(lambda x: f"{x:.2f}%")
            ht["Value (USD)"] = ht["value"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(ht[["name","ticker","currency","Weight (%)","Return (%)","Value (USD)"]]
                         .rename(columns={"name":"Company","ticker":"Ticker","currency":"Ccy"})
                         .sort_values("Return (%)", ascending=False),
                         width='stretch', hide_index=True)

    # =========================================================================
    # HOME — Portfolio Structure
    # =========================================================================
    elif home_tab == "Portfolio Structure Analysis":
        st.markdown("## **Portfolio Structure Analysis**")

        @st.cache_data(show_spinner=False)
        def fetch_countries(tickers: tuple) -> dict:
            SUFFIX_MAP = {
                ".NS":"India",".BO":"India",".DE":"Germany",".F":"Germany",
                ".SW":"Switzerland",".PA":"France",".AS":"Netherlands",".L":"United Kingdom",
                ".HK":"Hong Kong",".AX":"Australia",".TO":"Canada",".V":"Canada",
                ".T":"Japan",".KS":"South Korea",".SS":"China",".SZ":"China",
                ".BR":"Belgium",".ST":"Sweden",".CO":"Denmark",".OL":"Norway",
                ".HE":"Finland",".LS":"Portugal",".MC":"Spain",".MI":"Italy",
            }
            result = {}
            for t in tickers:
                try:
                    info = yf.Ticker(t).info
                    c    = info.get("country") or info.get("headquartersCountry")
                    if c: result[t] = c; continue
                except: pass
                for sfx, country in SUFFIX_MAP.items():
                    if t.upper().endswith(sfx.upper()):
                        result[t] = country; break
                else:
                    result[t] = "United States"
            return result

        REGION = {
            "India":"APAC","China":"APAC","Hong Kong":"APAC","Japan":"APAC",
            "South Korea":"APAC","Australia":"APAC","Singapore":"APAC","Taiwan":"APAC",
            "United States":"Americas","Canada":"Americas","Brazil":"Americas","Mexico":"Americas",
            "Germany":"EMEA","France":"EMEA","Switzerland":"EMEA","Netherlands":"EMEA",
            "United Kingdom":"EMEA","Sweden":"EMEA","Denmark":"EMEA","Norway":"EMEA",
            "Finland":"EMEA","Belgium":"EMEA","Spain":"EMEA","Italy":"EMEA",
            "Portugal":"EMEA","Ireland":"EMEA","Luxembourg":"EMEA","Austria":"EMEA",
            "Israel":"EMEA","South Africa":"EMEA","Saudi Arabia":"EMEA","UAE":"EMEA",
        }
        ISO = {
            "India":"IND","China":"CHN","Hong Kong":"HKG","Japan":"JPN","South Korea":"KOR",
            "Australia":"AUS","Singapore":"SGP","Taiwan":"TWN","United States":"USA",
            "Canada":"CAN","Brazil":"BRA","Mexico":"MEX","Germany":"DEU","France":"FRA",
            "Switzerland":"CHE","Netherlands":"NLD","United Kingdom":"GBR","Sweden":"SWE",
            "Denmark":"DNK","Norway":"NOR","Finland":"FIN","Belgium":"BEL","Spain":"ESP",
            "Italy":"ITA","Portugal":"PRT","Ireland":"IRL","Luxembourg":"LUX","Austria":"AUT",
            "Israel":"ISR","South Africa":"ZAF","Saudi Arabia":"SAU","UAE":"ARE",
        }

        with st.spinner("Fetching HQ countries…"):
            country_map = fetch_countries(tuple(sorted(portfolio.keys())))

        try:
            final_row    = nav_df.iloc[-1]
            final_nav    = final_row["nav_usd"]
            final_cash   = final_row["cash_usd"]
            final_equity = final_row["equity_usd"]
            final_date   = nav_df.index[-1]
            live_shares  = final_row["live_shares"]

            # Market cap from fast_info (already called above in price download session)
            def get_mc(ticker):
                try:
                    mc = getattr(yf.Ticker(ticker).fast_info,"market_cap",None) \
                         or yf.Ticker(ticker).info.get("marketCap") or 0
                    return mc
                except: return 0

            def classify_mc(mc):
                if mc >= 10e9: return "Large"
                elif mc >= 2e9: return "Mid"
                elif mc > 0:   return "Small"
                return "Unknown"

            rows = []
            for ticker, info in portfolio.items():
                p_nat = price_on(ticker, final_date, prices_df) or info["purchase_price"]
                val   = p_nat * spot_usd.get(info["currency"],1.0) * info["quantity"]
                country = country_map.get(ticker,"United States")
                rows.append({
                    "ticker": ticker, "name": info["name"],
                    "sector": info["sector"], "country": country,
                    "country_iso": ISO.get(country,""),
                    "region":  REGION.get(country,"Unknown"),
                    "market_cap": classify_mc(get_mc(ticker)),
                    "currency": info["currency"],
                    "value_usd": val,
                    "weight": val / final_nav * 100 if final_nav else 0,
                })

            df_an    = pd.DataFrame(rows)
            cash_pct = final_cash / final_nav * 100 if final_nav else 0
            eq_pct   = final_equity / final_nav * 100 if final_nav else 0

            # Sector pie
            st.markdown("### **Sector Distribution**")
            sec = df_an.groupby("sector")["weight"].sum().reset_index()
            sec.columns = ["Sector","Weight (%)"]
            if cash_pct > 0:
                sec = pd.concat([sec, pd.DataFrame([{"Sector":"Cash","Weight (%)":cash_pct}])], ignore_index=True)
            c1, c2 = st.columns([2,1])
            with c1:
                fig = px.pie(sec, values="Weight (%)", names="Sector",
                             title="Sector Allocation (% of NAV)",
                             color_discrete_sequence=BLUE_SCALE, hole=0.4)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, width='stretch')
            with c2:
                st.markdown("### Breakdown")
                for _, r in sec.sort_values("Weight (%)", ascending=False).iterrows():
                    st.metric(r["Sector"], f"{r['Weight (%)']:.1f}%")

            # Geo map
            st.markdown("### **Geographical Distribution**")
            c1, c2 = st.columns([2,1])
            with c1:
                cnt = df_an.groupby(["country","country_iso"])["weight"].sum().reset_index()
                cnt.columns = ["Country","ISO","Weight (%)"]
                fig = px.choropleth(cnt, locations="ISO", color="Weight (%)", hover_name="Country",
                                    color_continuous_scale=[[0,"#DBEAFE"],[0.5,"#3B82F6"],[1,NAVY]],
                                    title="Geographic Distribution (% of NAV)")
                fig.update_geos(showcountries=True, countrycolor="lightgray",
                                showcoastlines=True, projection_type="natural earth")
                fig.update_layout(height=450, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, width='stretch')
            with c2:
                st.markdown("### Regional")
                reg = df_an.groupby("region")["weight"].sum().reset_index()
                reg.columns = ["Region","Weight (%)"]
                for _, r in reg.sort_values("Weight (%)", ascending=False).iterrows():
                    st.metric(r["Region"], f"{r['Weight (%)']:.1f}%")
                st.dataframe(cnt[["Country","Weight (%)"]].sort_values("Weight (%)", ascending=False)
                             .style.format({"Weight (%)":"{:.1f}%"}), width='stretch', hide_index=True)

            # Market cap & currency
            st.markdown("### **Additional Analysis**")
            c1, c2 = st.columns(2)
            with c1:
                mc = df_an.groupby("market_cap")["weight"].sum().reset_index()
                mc.columns = ["Market Cap","Weight (%)"]
                fig = px.bar(mc, x="Market Cap", y="Weight (%)", title="Market Cap Distribution",
                             color="Weight (%)",
                             color_continuous_scale=[[0,"#DBEAFE"],[0.5,"#3B82F6"],[1,NAVY]])
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, width='stretch')
            with c2:
                cur = df_an.groupby("currency")["weight"].sum().reset_index()
                cur.columns = ["Currency","Weight (%)"]
                existing_usd = cur.loc[cur["Currency"]=="USD","Weight (%)"].sum()
                if existing_usd > 0:
                    cur.loc[cur["Currency"]=="USD","Weight (%)"] += cash_pct
                else:
                    cur = pd.concat([cur, pd.DataFrame([{"Currency":"USD (cash)","Weight (%)":cash_pct}])], ignore_index=True)
                fig = px.pie(cur, values="Weight (%)", names="Currency",
                             title="Currency Exposure incl. Cash",
                             color_discrete_sequence=BLUE_SCALE)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            # Concentration
            st.markdown("### **Concentration Metrics**")
            df_s  = df_an.sort_values("weight", ascending=False)
            top5  = df_s.head(5)["weight"].sum()
            top10 = df_s.head(10)["weight"].sum()
            hhi   = (df_an["weight"]**2).sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Top 5 Holdings",  f"{top5:.1f}%")
            c2.metric("Top 10 Holdings", f"{top10:.1f}%")
            with c3:
                st.metric("HHI Index", f"{hhi:.0f}")
                st.caption("✅ Well diversified" if hhi<1000 else "⚠️ Moderate" if hhi<1800 else "🔴 Concentrated")

            st.markdown("### **Top 10 Holdings**")
            th = df_s[["name","sector","country","currency","value_usd","weight"]].head(10).copy()
            th["value_usd"] = th["value_usd"].apply(lambda x: f"${x:,.0f}")
            th["weight"]    = th["weight"].apply(lambda x: f"{x:.2f}%")
            th.columns = ["Company","Sector","Country","Ccy","Value (USD)","Weight"]
            st.dataframe(th, width='stretch', hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")

    # =========================================================================
    # HOME — Forecast
    # =========================================================================
    elif home_tab == "Forecast":
        st.markdown("## **Forecast**")
        # Use prices_df which is already loaded — just take the last year of data
        one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
        hist_prices  = prices_df.loc[str(one_year_ago.date()):]

        forecast_method = st.tabs(["Monte Carlo Simulation","Analyst Consensus","DCF Analysis"])

        # ── Monte Carlo ──────────────────────────────────────────────────────
        with forecast_method[0]:
            st.markdown("### **Monte Carlo Simulation**")
            st.markdown("All values in USD.")

            c1, c2, c3 = st.columns(3)
            with c1: num_sim = st.number_input("Simulations",   100, 10000, 1000, 100)
            with c2: time_h  = st.number_input("Horizon (days)", 30,  1825,  252,  30)
            with c3:
                cur_val_usd = sum(
                    (current_price(t) or 0) * info["quantity"] * spot_usd.get(info["currency"],1.0)
                    for t, info in portfolio.items()
                )
                init_inv = st.number_input("Initial Value (USD)", 1000, value=int(cur_val_usd), step=1000)

            if st.button("Run Monte Carlo", type="primary"):
                pv   = {t: (current_price(t) or 0) * portfolio[t]["quantity"]
                            * spot_usd.get(portfolio[t]["currency"],1.0) for t in portfolio}
                tot  = sum(pv.values()) or 1
                wts  = {t: pv[t]/tot for t in pv}

                # Returns from prices_df (already downloaded, no extra calls)
                ret_df = hist_prices[[t for t in portfolio if t in hist_prices.columns]].pct_change().dropna()

                if ret_df.empty or len(ret_df) < 30:
                    st.error("Not enough history.")
                else:
                    mu  = ret_df.mean()
                    cov = ret_df.cov() + np.eye(len(ret_df.columns)) * 1e-8
                    np.random.seed(42)
                    tl   = list(ret_df.columns)
                    sims = np.zeros((time_h, num_sim))
                    for i in range(num_sim):
                        vals = [init_inv]
                        for _ in range(time_h):
                            try:    rr = mu.values + np.linalg.cholesky(cov) @ np.random.standard_normal(len(tl))
                            except: rr = np.random.normal(mu.values, np.sqrt(np.diag(cov)))
                            pr = sum(wts.get(tl[j],0) * rr[j] for j in range(len(tl)))
                            vals.append(vals[-1] * (1+pr))
                        sims[:,i] = vals[1:]

                    p5, p50, p95 = (np.percentile(sims, p, axis=1) for p in [5,50,95])
                    fig = go.Figure()
                    for i in range(0, num_sim, max(1, num_sim//100)):
                        fig.add_trace(go.Scatter(x=list(range(time_h)), y=sims[:,i], mode="lines",
                                                 line=dict(width=0.5, color="lightgray"),
                                                 showlegend=False, hoverinfo="skip"))
                    for arr, name, color in [(p5,"Bear (5%)","red"),(p50,"Base (50%)","blue"),(p95,"Bull (95%)","green")]:
                        fig.add_trace(go.Scatter(x=list(range(time_h)), y=arr, mode="lines",
                                                 name=name, line=dict(color=color, width=3)))
                    fig.update_layout(title=f"Monte Carlo: {num_sim} paths over {time_h} days",
                                      xaxis_title="Days", yaxis_title="Portfolio Value (USD)",
                                      height=500, hovermode="x unified")
                    st.plotly_chart(fig, width='stretch')
                    c1, c2, c3, c4 = st.columns(4)
                    for col, val, label in [(c1,p5[-1],"Bear"),(c2,p50[-1],"Base"),(c3,p95[-1],"Bull"),
                                            (c4,np.mean(sims[-1,:]),"Expected")]:
                        col.metric(label, f"${val:,.0f}", f"{(val-init_inv)/init_inv*100:.1f}%")

        # ── Analyst Consensus ────────────────────────────────────────────────
        with forecast_method[1]:
            st.markdown("### **Analyst Consensus Forecast**")
            rows = []
            for ticker, info in portfolio.items():
                try:
                    yf_info  = yf.Ticker(ticker).info
                    ccy      = info["currency"]
                    cp_nat   = current_price(ticker) or 0
                    cp_usd   = cp_nat * spot_usd.get(ccy,1.0)
                    tp_nat   = yf_info.get("targetMeanPrice")
                    if not tp_nat: continue
                    tp_usd   = tp_nat * spot_usd.get(ccy,1.0)
                    up       = (tp_usd - cp_usd) / cp_usd * 100 if cp_usd else 0
                    cv       = cp_usd * info["quantity"]
                    rows.append({
                        "Ticker": ticker, "Currency": ccy,
                        f"Current ({ccy})":  round(cp_nat,2),
                        f"Target ({ccy})":   round(tp_nat,2),
                        "Current (USD)": round(cp_usd,2), "Target (USD)": round(tp_usd,2),
                        "Upside (%)": round(up,2),
                        "Analysts": yf_info.get("numberOfAnalystOpinions",0),
                        "Rec": yf_info.get("recommendationKey","N/A").upper(),
                        "Value (USD)": cv, "Projected (USD)": cv*(1+up/100),
                    })
                except Exception as e:
                    st.warning(f"{ticker}: {e}")
            if rows:
                da  = pd.DataFrame(rows)
                tc, tp_, tg = da["Value (USD)"].sum(), da["Projected (USD)"].sum(), (da["Projected (USD)"]-da["Value (USD)"]).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Value (USD)", f"${tc:,.0f}")
                c2.metric("Projected (USD)",     f"${tp_:,.0f}", f"{tg/tc*100:.1f}%")
                c3.metric("Potential Gain",       f"${tg:,.0f}")
                st.dataframe(da, width='stretch', hide_index=True)
                fig = go.Figure(go.Bar(
                    x=da["Ticker"], y=da["Upside (%)"],
                    marker_color=["green" if x>0 else "red" for x in da["Upside (%)"]],
                    text=[f"{x:.1f}%" for x in da["Upside (%)"]],textposition="outside",
                ))
                fig.update_layout(title="Analyst Upside/Downside", height=400, showlegend=False)
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("No analyst data available.")

        # ── DCF ──────────────────────────────────────────────────────────────
        with forecast_method[2]:
            st.markdown("### **DCF Analysis**")
            sel = st.selectbox("Select Stock", list(portfolio.keys()))
            if sel:
                ccy_sel = portfolio[sel]["currency"]
                c1, c2  = st.columns(2)
                with c1:
                    try:
                        cf = yf.Ticker(sel).cashflow
                        fcf_def = int(abs(cf.loc["Free Cash Flow"].iloc[0])) \
                                  if not cf.empty and "Free Cash Flow" in cf.index else 1_000_000_000
                    except: fcf_def = 1_000_000_000
                    fcf_in = st.number_input(f"Current FCF ({ccy_sel})", 0, value=fcf_def, step=1_000_000)
                    gr     = st.slider("Growth Rate (%)", -10.0, 50.0, 5.0, 0.5)
                    py     = st.number_input("Projection years", 1, 10, 5)
                    wacc_  = st.slider("WACC (%)", 1.0, 20.0, 10.0, 0.5)
                    tg     = st.slider("Terminal Growth (%)", 0.0, 5.0, 2.5, 0.1)
                    try:    so = int(yf.Ticker(sel).info.get("sharesOutstanding",1_000_000_000))
                    except: so = 1_000_000_000
                    shares = st.number_input("Shares Outstanding", 1_000_000, value=so, step=1_000_000)
                with c2:
                    pf    = [fcf_in*(1+gr/100)**y for y in range(1,py+1)]
                    pv    = [pf[i]/(1+wacc_/100)**(i+1) for i in range(py)]
                    tv    = pf[-1]*(1+tg/100)/(wacc_/100-tg/100)
                    pv_tv = tv/(1+wacc_/100)**py
                    ev    = sum(pv)+pv_tv
                    fv    = ev/shares
                    cp    = current_price(sel) or 0
                    ud    = (fv-cp)/cp*100 if cp else 0

                    m1, m2 = st.columns(2)
                    m1.metric(f"Fair Value ({ccy_sel})", f"{fv:.2f}")
                    m1.metric("Fair Value (USD)",         f"${fv*spot_usd.get(ccy_sel,1.0):.2f}")
                    m1.metric("Enterprise Value",         f"{ccy_sel} {ev/1e9:.2f}B")
                    m2.metric(f"Current Price ({ccy_sel})", f"{cp:.2f}")
                    m2.metric("Current (USD)",             f"${cp*spot_usd.get(ccy_sel,1.0):.2f}")
                    m2.metric("Upside/Downside",           f"{ud:.1f}%", delta=f"{ud:.1f}%")
                    st.dataframe(pd.DataFrame([{"Year":y+1,
                        f"FCF ({ccy_sel})":f"{pf[y]/1e6:.1f}M",
                        "Discount Factor":f"{1/(1+wacc_/100)**(y+1):.4f}",
                        f"PV ({ccy_sel})":f"{pv[y]/1e6:.1f}M"} for y in range(py)]),
                        width='stretch', hide_index=True)
                    if ud>20:    st.success(f"Undervalued by {ud:.1f}%")
                    elif ud<-20: st.error(f"Overvalued by {abs(ud):.1f}%")
                    else:        st.info("Fairly valued (±20%)")


# =============================================================================
# SECTOR PAGES
# =============================================================================
elif main_page in ["TMT","FIG","Industrials","PUI","Consumer Goods","Healthcare","Real Estate"]:
    sector_name     = main_page
    sector_holdings = {k: v for k, v in portfolio.items() if v["sector"] == sector_name}

    st.title(f"{sector_name} — Sector Analysis")

    if not sector_holdings:
        st.warning(f"No holdings in '{sector_name}'. Check the 'sector' column in your Excel.")
    else:
        if "sector_tab" not in st.session_state:
            st.session_state.sector_tab = "Performance Analysis"

        c1, c2, c3 = st.columns(3)
        for col, label, key in [
            (c1,"Performance Analysis","perf"),
            (c2,"Financial Analysis","fin"),
            (c3,"Company Specific","spec"),
        ]:
            with col:
                if st.button(label, key=f"{key}_{sector_name}", width='stretch'):
                    st.session_state.sector_tab = label
                    st.rerun()

        st.markdown("---")
        sector_tab = st.session_state.sector_tab

        # ── Performance Analysis ─────────────────────────────────────────────
        if sector_tab == "Performance Analysis":
            st.markdown(f"## Performance Analysis — {sector_name}")

            bm_ticker = SECTOR_ETFS.get(sector_name, MSCI_WORLD)
            c1, c2    = st.columns(2)
            with c1: start_date = st.date_input("Start Date", pd.to_datetime("2025-11-06").date(), key=f"s_{sector_name}")
            with c2: end_date   = st.date_input("End Date",   pd.Timestamp.today().date(),          key=f"e_{sector_name}")

            if st.button("Generate Performance Analysis", type="primary", key=f"gen_{sector_name}"):
                if start_date >= end_date:
                    st.error("Start date must be before end date."); st.stop()

                bm_close = price_series(bm_ticker).loc[str(start_date):str(end_date)]
                if bm_close.empty:
                    st.error(f"No price data for benchmark {bm_ticker}."); st.stop()

                # Build daily sector portfolio value in USD from prices_df (no extra downloads)
                date_range = bm_close.index
                sec_nav    = nav_df.loc[str(start_date):str(end_date)]

                # Compute sector-specific value: sum(price_usd * shares_held) for sector tickers
                first_sector_trade = tx[tx["Ticker"].isin(sector_holdings.keys())]["Date"].min().normalize()
                active_dates = date_range[date_range >= first_sector_trade]

                sec_vals = pd.Series(dtype=float, index=active_dates)
                for date in active_dates:
                    live = nav_df.loc[date, "live_shares"] if date in nav_df.index else {}
                    sec_vals[date] = sum(
                        (price_on(t, date, prices_df) or 0) * live.get(t, 0) * get_fx_usd(portfolio[t]["currency"], date, fx_usd_df)
                        for t in sector_holdings if t in live
                    )
                sec_vals = sec_vals[sec_vals > 0]
                if sec_vals.empty:
                    st.warning("Could not compute sector values."); st.stop()

                from scipy import stats as scipy_stats

                sec_ret = sec_vals.pct_change().dropna()
                sec_ret = sec_ret[sec_ret.abs() <= 0.25]
                bm_ret  = bm_close.pct_change().dropna()
                common  = sec_ret.index.intersection(bm_ret.index)
                sr, br  = sec_ret.loc[common], bm_ret.loc[common]

                trs = ((sec_vals.iloc[-1]/sec_vals.iloc[0])-1)*100
                trb = ((bm_close.iloc[-1]/bm_close.iloc[0])-1)*100
                rf_d    = RF_ANNUAL/252
                exc     = sr - rf_d
                sharpe  = (exc.mean()/exc.std(ddof=1))*np.sqrt(252) if exc.std(ddof=1) else 0
                rd      = pd.DataFrame({"b":br-rf_d,"s":exc}).dropna()
                if len(rd)>2:
                    sl, ic, *_ = scipy_stats.linregress(rd["b"],rd["s"])
                    beta, alpha_a = sl, ic*252
                else:
                    beta = alpha_a = 0.0
                vol_s = sr.std(ddof=1)*np.sqrt(252)*100
                vol_b = br.std(ddof=1)*np.sqrt(252)*100
                cum   = (1+sr).cumprod()
                mdd   = ((cum-cum.expanding().max())/cum.expanding().max()).min()*100
                act   = sr-br
                te    = act.std(ddof=1)*np.sqrt(252)
                ir    = (act.mean()*252)/te if te else 0

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Return (Sector)",    f"{trs:.2f}%", delta=f"{trs-trb:.2f}% vs bm")
                c1.metric("Sharpe",             f"{sharpe:.2f}")
                c2.metric("Return (Benchmark)", f"{trb:.2f}%")
                c2.metric("Beta",               f"{beta:.2f}")
                c3.metric("Vol (Sector)",       f"{vol_s:.2f}%")
                c3.metric("Max Drawdown",       f"{mdd:.2f}%")
                c4.metric("Vol (Benchmark)",    f"{vol_b:.2f}%")
                c4.metric("Info Ratio",         f"{ir:.2f}")
                c1,c2 = st.columns(2)
                c1.metric("Alpha (Ann.)",   f"{alpha_a*100:.2f}%")
                c2.metric("Tracking Error", f"{te*100:.2f}%")

                sn = (sec_vals/sec_vals.iloc[0])*100
                bn = (bm_close/bm_close.iloc[0])*100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sn.index, y=sn, name=f"{sector_name} (USD)", mode="lines",
                                         line=dict(color=NAVY, width=2)))
                fig.add_trace(go.Scatter(x=bn.index, y=bn, name=f"{bm_ticker} benchmark", mode="lines",
                                         line=dict(color=RED, width=2)))
                fig.update_xaxes(rangeselector=dict(buttons=[
                    dict(count=1,label="1M",step="month",stepmode="backward"),
                    dict(count=6,label="6M",step="month",stepmode="backward"),
                    dict(count=1,label="1Y",step="year",stepmode="backward"),
                    dict(step="all",label="All"),
                ]))
                fig.update_layout(yaxis_title="Base = 100", xaxis_title="Date")
                st.plotly_chart(fig, width='stretch')

                # Per-holding breakdown
                st.markdown("### Holdings Breakdown (USD)")
                hp_rows = []
                for ticker, info in sector_holdings.items():
                    ps = price_series(ticker).loc[str(start_date):str(end_date)]
                    if ps.empty: continue
                    ip  = float(ps.iloc[0]); cp_ = float(ps.iloc[-1])
                    ret = (cp_/ip - 1)*100
                    iv  = ip  * get_fx_usd(info["currency"], ps.index[0], fx_usd_df)  * info["quantity"]
                    cv  = cp_ * get_fx_usd(info["currency"], ps.index[-1], fx_usd_df) * info["quantity"]
                    hp_rows.append({
                        "Ticker": ticker, "Name": info["name"], "Ccy": info["currency"],
                        "Return (%)": round(ret,2),
                        "Weight (%)": round(iv/sec_vals.iloc[0]*100, 2) if sec_vals.iloc[0] else 0,
                        "Contribution (%)": round((cv-iv)/sec_vals.iloc[0]*100, 2) if sec_vals.iloc[0] else 0,
                    })
                st.dataframe(pd.DataFrame(hp_rows).sort_values("Return (%)", ascending=False),
                             hide_index=True, width='stretch')

                # Correlation matrix
                if len(sector_holdings) > 1:
                    ret_dict = {}
                    for t, info in sector_holdings.items():
                        ps = price_series(t).loc[str(start_date):str(end_date)]
                        if not ps.empty: ret_dict[info["name"]] = ps.pct_change().dropna()
                    corr_m = pd.DataFrame(ret_dict).corr()
                    up_tri = corr_m.values[np.triu_indices_from(corr_m.values, k=1)]
                    c1, c2 = st.columns(2)
                    c1.metric("VaR (95%)",          f"{np.percentile(sr,5)*100:.2f}%")
                    c2.metric("Avg Correlation",    f"{up_tri.mean():.2f}")
                    st.dataframe(corr_m.style.background_gradient(cmap="coolwarm",vmin=-1,vmax=1).format("{:.2f}"),
                                 width='stretch')

        # ── Financial Analysis ───────────────────────────────────────────────
        elif sector_tab == "Financial Analysis":
            st.markdown(f"## Financial Analysis — {sector_name}")
            if st.button("Generate Financial Analysis", type="primary", key=f"gen_fin_{sector_name}"):
                fin_rows, failed = [], []
                ph = st.empty()
                for i, (ticker, info) in enumerate(sector_holdings.items()):
                    ph.info(f"Fetching {ticker} ({i+1}/{len(sector_holdings)})…")
                    ti, err = _yf_info_retry(ticker)
                    if err:
                        time.sleep(15)
                        ti, err2 = _yf_info_retry(ticker, max_attempts=3, base_delay=5.0)
                        if err2: failed.append(ticker); continue
                    cf_df = bs_df = None
                    try:
                        tk    = yf.Ticker(ticker)
                        cf_df = tk.cash_flow
                        bs_df = tk.balance_sheet
                    except: pass
                    row = {"Ticker":ticker,"Name":info["name"],"Currency":info["currency"]}
                    row.update(_extract_ratios(ti, cf_df, bs_df))
                    fin_rows.append(row)
                    time.sleep(1.0)
                ph.empty()

                if failed:
                    st.warning(f"Rate-limited, skipped: {', '.join(failed)}")
                if not fin_rows:
                    st.error("No data retrieved."); st.stop()

                fin_df = pd.DataFrame(fin_rows)
                st.success(f"Data for {len(fin_rows)}/{len(sector_holdings)} holdings.")

                with st.spinner("Fetching sector benchmarks…"):
                    ind_avgs = fetch_sector_avg(sector_name)

                ratio_cfg = [("P/E Ratio","Price-to-Earnings"),("P/B Ratio","Price-to-Book"),
                             ("Debt/Equity","Debt-to-Equity"),("OCF Ratio","OCF Ratio")]
                c1, c2 = st.columns(2)
                for idx, (col, title) in enumerate(ratio_cfg):
                    cd = fin_df[["Name",col]].dropna()
                    if cd.empty: (c1 if idx%2==0 else c2).caption(f"*{title}: no data*"); continue
                    ind_avg  = ind_avgs.get(col)
                    port_avg = cd[col].mean()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=cd["Name"],y=cd[col],name=title,
                                         marker_color="#030C30",text=cd[col].round(2),textposition="outside"))
                    if ind_avg:
                        fig.add_trace(go.Scatter(x=cd["Name"],y=[ind_avg]*len(cd),mode="lines",
                                                 name=f"Industry median: {ind_avg:.2f}",
                                                 line=dict(color=RED,width=2,dash="dot")))
                    if len(cd)>1:
                        fig.add_trace(go.Scatter(x=cd["Name"],y=[port_avg]*len(cd),mode="lines",
                                                 name=f"Portfolio avg: {port_avg:.2f}",
                                                 line=dict(color="#FFA500",width=1.5,dash="dashdot")))
                    fig.update_layout(title=title,height=400,showlegend=True,template="plotly_white")
                    (c1 if idx%2==0 else c2).plotly_chart(fig, width='stretch')

                st.markdown("### Full Summary")
                st.dataframe(fin_df.fillna("N/A"), hide_index=True, width='stretch')
                nd = fin_df.select_dtypes(include=[np.number])
                if not nd.empty:
                    st.markdown("### Sector Statistics")
                    st.dataframe(nd.describe().T[["mean","50%","min","max","std"]].rename(columns={"50%":"median"}),
                                 width='stretch')
                if ind_avgs:
                    st.markdown("### Industry Benchmark Medians")
                    st.dataframe(pd.DataFrame([{"Ratio":k,"Median":round(v,2)} for k,v in ind_avgs.items()]),
                                 hide_index=True, width='stretch')

        # ── Company Specific ─────────────────────────────────────────────────
        elif sector_tab == "Company Specific":
            st.markdown(f"## Company Specific — {sector_name}")
            comp_opts = {info["name"]: ticker for ticker, info in sector_holdings.items()}
            sel_name  = st.selectbox("Select Company", list(comp_opts.keys()))

            if sel_name:
                sel_ticker = comp_opts[sel_name]
                cinfo      = sector_holdings[sel_ticker]
                ccy        = cinfo["currency"]

                st.markdown(f"### {sel_name} ({sel_ticker}) — {ccy}")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Sector",   cinfo["sector"])
                c2.metric("Currency", ccy)
                c3.metric("Qty Held", f"{cinfo['quantity']:,}")
                c4.metric(f"Avg Buy ({ccy})", f"{cinfo['purchase_price']:.2f}")

                # Transaction history table + running position chart
                st.markdown("---")
                tx_stock = cinfo.get("_transactions", pd.DataFrame())
                if not tx_stock.empty:
                    d = tx_stock.copy()
                    d["Date"] = d["Date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(d[["Date","Ticker","Action","Quantity"]], width='stretch', hide_index=True)

                    running, dates_r, vals_r = 0, [], []
                    for _, r in tx_stock.sort_values("Date").iterrows():
                        running += r["Quantity"] if r["Action"]=="Buy" else -r["Quantity"]
                        dates_r.append(r["Date"]); vals_r.append(running)
                    fig = go.Figure(go.Scatter(x=dates_r, y=vals_r, mode="lines+markers",
                                              line=dict(color=NAVY,width=2), marker=dict(size=8)))
                    fig.update_layout(title="Running Net Position",height=300,template="plotly_white")
                    st.plotly_chart(fig, width='stretch')

                # Price chart — reads directly from prices_df, no extra download
                st.markdown("---")
                st.markdown(f"### Price History ({ccy})")
                pd_dt = pd.to_datetime(cinfo["purchase_date"])
                c1, c2 = st.columns(2)
                with c1: chart_start = st.date_input("Start", (pd_dt-pd.Timedelta(days=30)).date(), key=f"cs2_{sel_ticker}")
                with c2: chart_end   = st.date_input("End",   pd.Timestamp.today().date(),           key=f"ce2_{sel_ticker}")

                try:
                    tkinfo    = yf.Ticker(sel_ticker).info
                    an_target = tkinfo.get("targetMeanPrice")
                    an_count  = tkinfo.get("numberOfAnalystOpinions",0)
                    an_rec    = tkinfo.get("recommendationKey","N/A").upper()
                except: an_target = None; an_count = 0; an_rec = "N/A"

                if st.button("Generate Stock Analysis", type="primary", key=f"gsa_{sel_ticker}"):
                    cl = price_series(sel_ticker).loc[str(chart_start):str(chart_end)]
                    if cl.empty:
                        st.error("No price data in that range.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=cl.index, y=cl, mode="lines",
                                                 name=f"{sel_ticker} ({ccy})",
                                                 line=dict(color=NAVY,width=2)))

                        # Buy / sell markers from transaction history
                        if not tx_stock.empty:
                            for _, tr in tx_stock.iterrows():
                                td = tr["Date"]
                                if chart_start <= td.date() <= chart_end:
                                    avail = cl.index[cl.index <= td]
                                    if len(avail):
                                        color  = "green" if tr["Action"]=="Buy" else "red"
                                        symbol = "triangle-up" if tr["Action"]=="Buy" else "triangle-down"
                                        fig.add_trace(go.Scatter(
                                            x=[td], y=[float(cl.loc[avail[-1]])], mode="markers",
                                            marker=dict(color=color,size=12,symbol=symbol),
                                            name=f"{tr['Action']} {int(tr['Quantity'])} @ {td.strftime('%Y-%m-%d')}",
                                        ))

                        target_price = cinfo.get("Target_price")
                        if target_price and target_price > 0:
                            fig.add_shape(type="line",x0=0,x1=1,xref="paper",
                                          y0=target_price,y1=target_price,
                                          line=dict(color="red",width=2,dash="dash"))
                            fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
                                                     name=f"Target {target_price:.2f} {ccy}",
                                                     line=dict(color="red",width=2,dash="dash"),showlegend=True))
                        fig.update_layout(title=f"{sel_name} Price History",
                                          xaxis_title="Date",yaxis_title=f"Price ({ccy})",
                                          hovermode="x unified",height=520,template="plotly_white")
                        fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=[
                            dict(count=1,label="1M",step="month",stepmode="backward"),
                            dict(count=3,label="3M",step="month",stepmode="backward"),
                            dict(count=6,label="6M",step="month",stepmode="backward"),
                            dict(count=1,label="1Y",step="year",stepmode="backward"),
                            dict(step="all",label="All"),
                        ]))
                        st.plotly_chart(fig, width='stretch')

                        cur_p   = float(cl.iloc[-1])
                        pp      = cinfo["purchase_price"]
                        tr_pct  = (cur_p-pp)/pp*100 if pp else 0
                        fx_spot = spot_usd.get(ccy,1.0)

                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric(f"Price ({ccy})",        f"{cur_p:.2f}",                delta=f"{tr_pct:.2f}% vs avg buy")
                        c1.metric("Price (USD)",            f"${cur_p*fx_spot:.2f}")
                        c2.metric(f"Position ({ccy})",     f"{cur_p*cinfo['quantity']:,.2f}")
                        c2.metric("Position (USD)",         f"${cur_p*cinfo['quantity']*fx_spot:,.0f}")
                        c3.metric(f"Gain/Loss ({ccy})",    f"{(cur_p-pp)*cinfo['quantity']:,.2f}", delta=f"{tr_pct:.2f}%")
                        c3.metric("Gain/Loss (USD)",        f"${(cur_p-pp)*cinfo['quantity']*fx_spot:,.0f}")
                        if target_price and target_price > 0:
                            c4.metric("Upside to Target", f"{(target_price-cur_p)/cur_p*100:.2f}%")
                        else:
                            c4.metric("Upside to Target", "N/A")

                        st.markdown("---")
                        st.markdown("### Price Targets")
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric(f"Your Target ({ccy})", f"{target_price:.2f}" if target_price else "N/A")
                        c2.metric("Analyst Consensus",    f"{an_target:.2f}" if an_target else "N/A")
                        c3.metric("No. Analysts",         an_count or "N/A")
                        c4.metric("Recommendation",       an_rec)

                        st.markdown("---")
                        st.markdown("### Investment Thesis")
                        st.info(cinfo.get("thesis","No thesis available."))

                        st.markdown("---")
                        st.markdown("### DCF Parameters")
                        c1, c2 = st.columns(2)
                        c1.write(f"**WACC:** {cinfo.get('WACC','N/A')}%")
                        with c2:
                            st.write(f"**Cash Flow Projections ({ccy}):**")
                            for i in range(1,6):
                                st.write(f"Year {i}: {cinfo.get(f'CF_{i}','N/A')}")

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.info("**HIC Capital Dashboard v5**\n\nall prices fetched once instead of on each request.")
