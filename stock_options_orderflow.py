# === STOCK OPTIONS RANKER v1.0 (SHADOW MODE ONLY) ===
# New engine for NSE F&O stock options.
# No order placement functions are included in this version.
#
# Architecture:
#   F&O universe -> liquidity/volatility filter -> market regime
#   -> setup detection -> 100-point ranking -> option liquidity
#   -> structural SL/2R target -> one-lot risk feasibility
#   -> shadow trade -> audit logs
#
# Credentials are read from the same .env.orderflow used by the existing bot:
#   OF_API_KEY
#   OF_ACCESS_TOKEN

import csv
import datetime as dt
import json
import logging
import math
import os
import threading
import time
import uuid
import concurrent.futures

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from kiteconnect import KiteConnect


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

load_dotenv(".env.orderflow")

API_KEY = os.getenv("OF_API_KEY")
ACCESS_TOKEN = os.getenv("OF_ACCESS_TOKEN")

if not API_KEY or not ACCESS_TOKEN:
    raise SystemExit("OF_API_KEY or OF_ACCESS_TOKEN not found in .env.orderflow")

PORT = 8065
STRATEGY_VERSION = "stock-options-v2.0-no-breakout-shadow"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

STATE_FILE = "stock_options_orderflow_state.json"

# Universe
MIN_PRICE = 100.0
MIN_AVG_TURNOVER_CRORE = 100.0
MIN_ATR_PCT = 1.20
MAX_ATR_PCT = 6.00
MAX_UNIVERSE_SIZE = 40

# Intraday
SCAN_START = dt.time(9, 30)
ENTRY_START = dt.time(10, 0)
ENTRY_CUTOFF = dt.time(15, 00)
HARD_EXIT = dt.time(15, 15)

MIN_RVOL = 1.00
PREFERRED_RVOL = 1.30
MAX_UNDERLYING_SPREAD_PCT = 0.10

# Option liquidity
MIN_OPTION_OI = 25_000
MIN_OPTION_VOLUME = 5_000
MAX_OPTION_SPREAD_PCT = 2.0
PREFERRED_OPTION_SPREAD_PCT = 1.0
MIN_OPTION_PREMIUM = 20.0
MAX_OPTION_PREMIUM = 250.0
MIN_DTE = 2
MAX_DTE = 10
AVOID_EXPIRY_DAY = True

# Risk / selection
MAX_RISK_PER_TRADE = 2_000.0       # CHANGE BEFORE LIVE TRADING
MAX_TRADES_PER_DAY = 3
MAX_DAILY_RISK = 5_000.0           # ceiling on total ₹ risk deployed in a day, CHANGE BEFORE LIVE TRADING
ONE_TRADE_PER_STOCK_PER_DAY = True
MIN_RANK_SCORE = 68.0
SECOND_TRADE_MIN_SCORE = 78.0
MIN_SCORE_GAP = 4.0
TARGET_R_MULTIPLE = 2.0            # v1 starts at fixed 1:2

# Dynamic option-premium profit protection
PROFIT_LOCK_1_R = 1.00       # activate first protection at +1R
PROFIT_LOCK_1_LOCK_R = 0.25 # protect +0.25R

PROFIT_LOCK_2_R = 1.50       # stronger protection at +1.5R
PROFIT_LOCK_2_LOCK_R = 0.75  # protect +0.75R

MAX_ENTRY_EXTENSION_ATR = 0.30

MAX_SL_ATR = 1.50
MIN_SL_ATR = 0.15
INTRADAY_CACHE_SECONDS = 60

# We estimate one-lot option risk using delta, then add a conservative
# uncertainty buffer. Actual shadow P&L is always marked from option quotes.
DELTA_RISK_BUFFER = 1.25

UNIVERSE_REFRESH_SECONDS = 15 * 60
CANDIDATE_REFRESH_SECONDS = 60
ACTIVE_TRADE_REFRESH_SECONDS = 3
HEARTBEAT_SECONDS = 30

IST = pytz.timezone("Asia/Kolkata")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

current_state = {
    "decision": "NO TRADE",
    "reason": "Initializing...",
    "strategy_version": STRATEGY_VERSION,
    "shadow_mode": True,
    "last_scan": None,
    "market_regime": "UNKNOWN",
    "market_score": 0,
    "nifty": {},
    "vix": {},
    "universe_count": 0,
    "filtered_count": 0,
    "candidate_count": 0,
    "top_candidates": [],
    "rejections": [],
    "active_trade": None,
    "daily_trades": 0,
    "daily_pnl": 0.0,
}

active_trade = None
daily_trades = 0
daily_pnl = 0.0
daily_risk_deployed = 0.0
daily_reset_date = None
traded_stocks_today = set()

nfo_df = pd.DataFrame()
stock_universe = {}
candidate_cache = {}
intraday_cache = {}
pending_pullbacks = {}
last_universe_refresh = 0.0
last_candidate_refresh = 0.0
last_emit = 0.0
last_emitted_state = None


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def now_ist():
    return dt.datetime.now(IST).replace(tzinfo=None)


def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if obj is None:
        return None

    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    return obj


def clean_dict(d):
    return {k: json_safe(v) for k, v in d.items()}


def kite_call(func, *args, timeout=8, **kwargs):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except Exception as exc:
        logging.warning("Kite call failed: %s | %s", getattr(func, "__name__", func), exc)
        try:
            future.cancel()
        except Exception:
            pass
        return None
    finally:
        executor.shutdown(wait=False)


def append_csv(path, row):
    row = clean_dict(row)
    try:
        exists = os.path.exists(path)
        if exists:
            with open(path, "r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
        else:
            header = []

        if not header:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            return

        fields = list(header) + [k for k in row if k not in header]
        if fields != header:
            old = pd.read_csv(path)
            for col in fields:
                if col not in old:
                    old[col] = ""
            old = old[fields]
            old.to_csv(path, index=False)
            header = fields

        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header).writerow(
                {k: row.get(k, "") for k in header}
            )
    except Exception as exc:
        logging.warning("CSV append failed %s: %s", path, exc)


def log_event(event, **data):
    path = os.path.join(LOG_DIR, "stock_options_orderflow.jsonl")
    payload = {"timestamp": now_ist().isoformat(), "event": event}
    payload.update(clean_dict(data))
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:
        logging.warning("JSON log failed: %s", exc)


def emit_state(force=False):
    global last_emit, last_emitted_state
    payload = clean_dict(current_state)
    if not force and payload == last_emitted_state and time.time() - last_emit < HEARTBEAT_SECONDS:
        return
    socketio.emit("stock_options_signal", payload)
    last_emit = time.time()
    last_emitted_state = json.loads(json.dumps(payload, default=str))


def reset_day_if_needed():
    global daily_reset_date, daily_trades, daily_pnl, daily_risk_deployed, traded_stocks_today
    today = now_ist().date()
    if daily_reset_date != today:
        daily_reset_date = today
        daily_trades = 0
        daily_pnl = 0.0
        daily_risk_deployed = 0.0
        traded_stocks_today = set()
        save_state()


def market_hours():
    t = now_ist().time()
    return dt.time(9, 15) <= t <= HARD_EXIT


# ---------------------------------------------------------------------------
# PERSISTENCE
# ---------------------------------------------------------------------------

def save_state():
    data = {
        "date": daily_reset_date.isoformat() if daily_reset_date else None,
        "daily_trades": daily_trades,
        "daily_pnl": daily_pnl,
        "daily_risk_deployed": daily_risk_deployed,
        "traded_stocks_today": list(traded_stocks_today),
        "active_trade": active_trade,
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2)
    except Exception as exc:
        logging.warning("State save failed: %s", exc)


def load_state():
    global daily_reset_date, daily_trades, daily_pnl, daily_risk_deployed, traded_stocks_today, active_trade
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        stored_date = data.get("date")
        today = now_ist().date().isoformat()
        if stored_date == today:
            daily_reset_date = now_ist().date()
            daily_trades = int(data.get("daily_trades", 0))
            daily_pnl = float(data.get("daily_pnl", 0))
            daily_risk_deployed = float(data.get("daily_risk_deployed", 0))
            traded_stocks_today = set(data.get("traded_stocks_today", []))
            # Do not automatically resurrect a shadow trade after a restart,
            # but record that it happened so the dangling ENTRY row can be
            # identified and excluded from P&L/edge analysis.
            orphaned = data.get("active_trade")
            if orphaned:
                log_event("shadow_trade_orphaned_on_restart", **orphaned)
            active_trade = None
        else:
            reset_day_if_needed()
    except Exception as exc:
        logging.warning("State load failed: %s", exc)


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

def get_quote(symbols):
    if not symbols:
        return {}
    data = kite_call(kite.quote, symbols, timeout=10)
    return data or {}


def get_historical(token, interval, days=5):
    end = now_ist()
    start = end - dt.timedelta(days=days)
    data = kite_call(
        kite.historical_data,
        int(token),
        start,
        end,
        interval,
        False,
        timeout=10,
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass
    return df


def atr(df, period=14):
    if df.empty or len(df) < period + 1:
        return np.nan
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def vwap(df):
    if df.empty:
        return np.nan
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    value = (typical * vol).cumsum()
    volume = vol.cumsum()
    out = value / volume
    return float(out.iloc[-1]) if not pd.isna(out.iloc[-1]) else np.nan


def rvol_time_aligned(df):
    if df.empty or len(df) < 20:
        return 1.0

    x = df.copy()
    x["slot"] = x["date"].dt.strftime("%H:%M")
    today = x["date"].dt.date.max()
    today_rows = x[x["date"].dt.date == today]
    if today_rows.empty:
        return 1.0

    latest = today_rows.iloc[-1]
    hist = x[(x["date"].dt.date != today) & (x["slot"] == latest["slot"])]
    if len(hist) >= 3 and hist["volume"].mean() > 0:
        return float(latest["volume"] / hist["volume"].mean())

    prior = x[x["date"].dt.date != today]["volume"].tail(20)
    if len(prior) and prior.mean() > 0:
        return float(latest["volume"] / prior.mean())
    return 1.0


def adx(df, period=14):
    if df.empty or len(df) < period * 2:
        return 0.0
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    value = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(0 if pd.isna(value) else value)


def rsi(df, period=14):
    if len(df) < period + 1:
        return 50.0
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    out = value.iloc[-1]
    return float(50 if pd.isna(out) else out)


def rolling_vwap_session(df):
    if df.empty:
        return np.nan
    today = df["date"].dt.date.max()
    return vwap(df[df["date"].dt.date == today])


# ---------------------------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------------------------

def load_nfo_instruments():
    global nfo_df
    raw = kite_call(kite.instruments, "NFO", timeout=20)
    if not raw:
        raise RuntimeError("Could not load NFO instruments")
    nfo_df = pd.DataFrame(raw)
    return nfo_df


def build_fno_universe():
    global stock_universe
    if nfo_df.empty:
        load_nfo_instruments()

    x = nfo_df.copy()
    if "segment" in x:
        fut = x[(x["segment"] == "NFO-FUT") & (x["instrument_type"] == "FUT")].copy()
    else:
        fut = x[x["instrument_type"] == "FUT"].copy()

    if fut.empty:
        raise RuntimeError("No NFO futures found")

    today = now_ist().date()
    fut["expiry"] = pd.to_datetime(fut["expiry"]).dt.date
    fut = fut[fut["expiry"] >= today].sort_values(["tradingsymbol", "expiry"])
    fut = fut.groupby("name", as_index=False).first()

    nse = kite_call(kite.instruments, "NSE", timeout=20)
    if not nse:
        raise RuntimeError("Could not load NSE instruments")
    nse_df = pd.DataFrame(nse)
    nse_map = {
        str(r["tradingsymbol"]): r
        for _, r in nse_df.iterrows()
        if r.get("instrument_type") == "EQ"
    }

    universe = {}
    for _, row in fut.iterrows():
        symbol = str(row.get("name", "")).strip()
        if symbol not in nse_map:
            continue
        eq = nse_map[symbol]
        universe[symbol] = {
            "symbol": symbol,
            "nse_token": int(eq["instrument_token"]),
            "future_token": int(row["instrument_token"]),
            "future_symbol": row["tradingsymbol"],
            "lot_size": int(row["lot_size"]),
        }

    stock_universe = universe
    logging.info("F&O stock universe built: %d", len(stock_universe))
    return universe


def daily_liquidity_profile(symbol, meta):
    df = get_historical(meta["nse_token"], "day", 35)
    if df.empty or len(df) < 20:
        return None, "insufficient daily history"

    df = df.tail(25)
    close = float(df["close"].iloc[-1])
    turnover = df["close"] * df["volume"]
    avg_turnover = float(turnover.tail(20).mean() / 1e7)
    a = atr(df, 14)
    atr_pct = float(a / close * 100) if close > 0 and not pd.isna(a) else 0.0

    if close < MIN_PRICE:
        return None, f"price {close:.1f} < {MIN_PRICE}"
    if avg_turnover < MIN_AVG_TURNOVER_CRORE:
        return None, f"20D turnover ₹{avg_turnover:.1f}cr < {MIN_AVG_TURNOVER_CRORE}cr"
    if atr_pct < MIN_ATR_PCT:
        return None, f"ATR {atr_pct:.2f}% < {MIN_ATR_PCT}%"
    if atr_pct > MAX_ATR_PCT:
        return None, f"ATR {atr_pct:.2f}% > {MAX_ATR_PCT}%"

    return {
        **meta,
        "price": close,
        "avg_turnover_cr": avg_turnover,
        "atr": a,
        "atr_pct": atr_pct,
    }, "PASS"


def refresh_universe(force=False):
    global last_universe_refresh, stock_universe
    if not force and time.time() - last_universe_refresh < UNIVERSE_REFRESH_SECONDS:
        return

    try:
        # Rebuild from the complete F&O universe on every scheduled
        # universe refresh. Do not progressively shrink the universe.
        build_fno_universe()

        passed = {}
        rejected = []
        raw_count = len(stock_universe)

        for symbol, meta in list(stock_universe.items()):
            profile, reason = daily_liquidity_profile(symbol, meta)
            if profile:
                passed[symbol] = profile
            else:
                rejected.append({"symbol": symbol, "reason": reason})
            time.sleep(0.45)  # ~2.2 req/sec ceiling on Kite's historical-data endpoint

        quote_symbols = [f"NSE:{s}" for s in passed]
        quotes = get_quote(quote_symbols)
        final = {}
        for symbol, profile in passed.items():
            q = quotes.get(f"NSE:{symbol}", {})
            ltp = float(q.get("last_price", profile["price"]) or profile["price"])
            volume = float(q.get("volume", 0) or 0)
            depth = q.get("depth", {}) or {}
            bid = float((depth.get("buy") or [{}])[0].get("price", 0) or 0)
            ask = float((depth.get("sell") or [{}])[0].get("price", 0) or 0)
            spread_pct = ((ask - bid) / ((ask + bid) / 2) * 100) if bid > 0 and ask > 0 else 999.0
            profile["ltp"] = ltp
            profile["today_volume"] = volume
            profile["underlying_bid"] = bid
            profile["underlying_ask"] = ask
            profile["underlying_spread_pct"] = spread_pct
            final[symbol] = profile

        final = dict(sorted(
            final.items(),
            key=lambda kv: kv[1]["avg_turnover_cr"],
            reverse=True
        )[:MAX_UNIVERSE_SIZE])

        stock_universe = final
        current_state["universe_count"] = raw_count
        current_state["filtered_count"] = len(final)
        current_state["universe_rejections"] = rejected[:100]

        append_csv(
            os.path.join(LOG_DIR, "stock_universe_filter.csv"),
            {
                "timestamp": now_ist(),
                "raw_fno_count": raw_count,
                "passed_count": len(final),
                "rejected_count": len(rejected),
            },
        )
        log_event("universe_refresh",
                  passed=len(final),
                  rejected=len(rejected),
                  raw=raw_count)
        last_universe_refresh = time.time()
    except Exception as exc:
        logging.exception("Universe refresh failed")
        current_state["reason"] = f"Universe refresh failed: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# MARKET REGIME
# ---------------------------------------------------------------------------

def get_market_context():
    quote = get_quote(["NSE:NIFTY 50", "NSE:INDIA VIX"])
    nq = quote.get("NSE:NIFTY 50", {})
    vq = quote.get("NSE:INDIA VIX", {})
    nifty = float(nq.get("last_price", 0) or 0)
    vix = float(vq.get("last_price", 0) or 0)

    nf = get_historical(256265, "5minute", 3)

    if nf.empty or nf["date"].dt.date.max() != now_ist().date():
        return {
            "regime": "STALE_DATA",
            "score": 0,
            "nifty": {"ltp": nifty},
            "vix": {"ltp": vix},
        }

    nvwap = rolling_vwap_session(nf)
    a = atr(nf, 14)
    price = float(nf["close"].iloc[-1])
    adx_val = adx(nf)
    ema9 = float(nf["close"].ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(nf["close"].ewm(span=21, adjust=False).mean().iloc[-1])

    # NIFTY index volume is not reliable for VWAP-based
    # relative-strength measurement.
    # Use today's opening price as the benchmark instead.
    today_nf = nf[nf["date"].dt.date == now_ist().date()]

    if not today_nf.empty:
        nifty_open = float(today_nf["open"].iloc[0])
    else:
        nifty_open = price

    nifty_intraday_return = (
        ((nifty / nifty_open) - 1) * 100
        if nifty_open > 0
        else 0.0
    )

    directional = abs(price - nvwap) / a if a and not pd.isna(a) and a > 0 else 0
    bullish = price > nvwap and ema9 > ema21
    bearish = price < nvwap and ema9 < ema21

    if adx_val >= 22 and directional >= 0.30:
        regime = "TRENDING_BULL" if bullish else "TRENDING_BEAR" if bearish else "MIXED"
        score = 8 if bullish or bearish else 4
    else:
        regime = "CHOPPY"
        score = 2

    if vix > 22:
        regime += "_HIGH_VIX"
    elif vix < 12:
        regime += "_LOW_VIX"

    return {
        "regime": regime,
        "score": score,
        "nifty": {
            "ltp": nifty,
            "vwap": nvwap,
            "open": nifty_open,
            "intraday_return": nifty_intraday_return,
            "adx": adx_val,
            "ema9": ema9,
            "ema21": ema21,
        },
        "vix": {"ltp": vix},
    }


# ---------------------------------------------------------------------------
# STOCK SETUP / RANKING
# ---------------------------------------------------------------------------

def get_stock_intraday(symbol, meta):
    # Historical candles are cached for 5 minutes. This is deliberate:
    # pulling 5m candles for 40 stocks every 60 seconds would unnecessarily
    # hammer the Kite historical-data endpoint.
    cached = intraday_cache.get(symbol)
    if cached and time.time() - cached["time"] < 60:
        return cached["df"]

    df5 = get_historical(meta["nse_token"], "5minute", 10)
    if df5.empty or len(df5) < 25:
        return None

    intraday_cache[symbol] = {"time": time.time(), "df": df5}
    return df5


def previous_day_levels(df):
    dates = df["date"].dt.date
    days = sorted(set(dates))
    if len(days) < 2:
        return None, None
    prev = df[dates == days[-2]]
    if prev.empty:
        return None, None
    return float(prev["high"].max()), float(prev["low"].min())


def detect_setup(symbol, meta, market):
    if meta.get("underlying_spread_pct", 999) > MAX_UNDERLYING_SPREAD_PCT:
        return None, f"underlying spread {meta.get('underlying_spread_pct', 999):.3f}% > {MAX_UNDERLYING_SPREAD_PCT}%"

    df = get_stock_intraday(symbol, meta)
    if df is None:
        return None, "intraday data unavailable"

    price = float(df["close"].iloc[-1])
    a = atr(df, 14)
    if pd.isna(a) or a <= 0:
        return None, "ATR unavailable"

    svwap = rolling_vwap_session(df)
    rvol = rvol_time_aligned(df)
    adx_val = adx(df)
    rsi_val = rsi(df)

    ema9 = float(df["close"].ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(df["close"].ewm(span=21, adjust=False).mean().iloc[-1])

    prev_high, prev_low = previous_day_levels(df)

    recent = df.tail(8)
    prior = df.iloc[-9:-1]
    if len(prior) < 5:
        return None, "not enough setup history"

    swing_high = float(prior["high"].max())
    swing_low = float(prior["low"].min())

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # V2: breakout is diagnostic only.
    # It is NOT required for entry.
    bull_break = price > swing_high and last["close"] > last["open"]
    bear_break = price < swing_low and last["close"] < last["open"]

    # V2: determine direction directly from the underlying setup.
    bullish_structure = (
            price > svwap
            and ema9 > ema21
            and price > prev["close"]
    )

    bearish_structure = (
            price < svwap
            and ema9 < ema21
            and price < prev["close"]
    )

    if bullish_structure:
        bias = "CALL"
    elif bearish_structure:
        bias = "PUT"
    else:
        return None, "no directional trend/structure"

    # Breakout is retained only for diagnostics.
    trigger = price



    score = 0.0
    components = {}

    trend = 0
    if bias == "CALL":
        if price > svwap: trend += 6
        if ema9 > ema21: trend += 5
        if market["regime"].startswith("TRENDING_BULL"): trend += 5
        if price > prev["close"]: trend += 4
    else:
        if price < svwap: trend += 6
        if ema9 < ema21: trend += 5
        if market["regime"].startswith("TRENDING_BEAR"): trend += 5
        if price < prev["close"]: trend += 4
    components["trend_structure"] = trend
    score += trend

    # ---------------------------------------------------------
    # V2 PARTICIPATION / CANDLE QUALITY
    #
    # Breakout quality is no longer part of the alpha score.
    # Range and volume expansion are retained as diagnostics only.
    # ---------------------------------------------------------

    candle_range = max(float(last["high"] - last["low"]), 1e-9)

    recent_ranges = (
        (prior["high"] - prior["low"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    median_range = (
        float(recent_ranges.tail(6).median())
        if len(recent_ranges) >= 3
        else candle_range
    )

    range_expansion = (
        candle_range / median_range
        if median_range > 0
        else 1.0
    )

    recent_volumes = (
        prior["volume"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    median_volume = (
        float(recent_volumes.tail(6).median())
        if len(recent_volumes) >= 3
        else 0.0
    )

    volume_expansion = (
        float(last["volume"]) / median_volume
        if median_volume > 0
        else 1.0
    )

    close_location = (
        (last["close"] - last["low"]) / candle_range
        if bias == "CALL"
        else (last["high"] - last["close"]) / candle_range
    )

    components["breakout_quality"] = 0
    components["range_expansion"] = round(range_expansion, 2)
    components["volume_expansion"] = round(volume_expansion, 2)
    components["close_location"] = round(close_location, 2)

    participation = 0
    if rvol >= 1.0: participation += 7
    if rvol >= 1.3: participation += 5
    if rvol >= 1.7: participation += 4
    components["rvol"] = participation
    score += participation

    # ---------------------------------------------------------
    # RELATIVE STRENGTH VS NIFTY
    # ---------------------------------------------------------
    # Compare the stock's intraday return from today's open
    # against NIFTY's intraday return from today's open.

    nifty_return = market["nifty"].get("intraday_return", 0.0)

    today_stock = df[df["date"].dt.date == now_ist().date()]

    if not today_stock.empty:
        stock_open = float(today_stock["open"].iloc[0])
    else:
        stock_open = price

    stock_return = (
        ((price / stock_open) - 1) * 100
        if stock_open > 0
        else 0.0
    )

    relative = stock_return - nifty_return

    rs_points = 0

    if bias == "CALL":
        if relative > 0.25:
            rs_points += 9
        if relative > 0.50:
            rs_points += 4
        if relative > 0.90:
            rs_points += 3
    else:
        if relative < -0.25:
            rs_points += 9
        if relative < -0.50:
            rs_points += 4
        if relative < -0.90:
            rs_points += 3

    components["relative_strength"] = rs_points
    score += rs_points

    momentum = 0
    if adx_val >= 18: momentum += 3
    if adx_val >= 25: momentum += 3
    if bias == "CALL" and 52 <= rsi_val <= 72: momentum += 3
    if bias == "PUT" and 28 <= rsi_val <= 48: momentum += 3
    if (bias == "CALL" and price > ema9) or (bias == "PUT" and price < ema9):
        momentum += 3
    components["momentum"] = momentum
    score += momentum

    alignment = 0
    if bias == "CALL" and market["regime"].startswith("TRENDING_BULL"):
        alignment = 10
    elif bias == "PUT" and market["regime"].startswith("TRENDING_BEAR"):
        alignment = 10
    elif market["regime"].startswith("CHOPPY"):
        alignment = 2
    else:
        alignment = 5
    components["market_alignment"] = alignment
    score += alignment

    penalties = 0
    if rvol < MIN_RVOL:
        penalties += 12
    if bias == "CALL" and rsi_val > 78:
        penalties += 8
    if bias == "PUT" and rsi_val < 22:
        penalties += 8
    if adx_val < 15:
        penalties += 8
    components["penalties"] = -penalties
    score -= penalties

    if bias == "CALL":
        structural_sl = min(
            float(recent["low"].iloc[-2]),
            float(last["low"])
        )

    else:
        structural_sl = max(
            float(recent["high"].iloc[-2]),
            float(last["high"])
        )

    buffer = 0.10 * a
    if bias == "CALL":
        sl = structural_sl - buffer
        risk_points = price - sl
        target = price + TARGET_R_MULTIPLE * risk_points
    else:
        sl = structural_sl + buffer
        risk_points = sl - price
        target = price - TARGET_R_MULTIPLE * risk_points

    # V2: anti-chase protection is based on distance from EMA9,
    # not distance from a breakout trigger.
    if bias == "CALL":
        extension = max(0, price - ema9) / a
    else:
        extension = max(0, ema9 - price) / a

    if risk_points <= 0:
        return None, "invalid structural risk"
    if risk_points < MIN_SL_ATR * a:
        return None, f"SL too tight ({risk_points/a:.2f} ATR)"
    if risk_points > MAX_SL_ATR * a:
        return None, f"SL too wide ({risk_points/a:.2f} ATR)"
    if extension > MAX_ENTRY_EXTENSION_ATR:
        return None, f"entry too extended from EMA9 ({extension:.2f} ATR)"

    if rvol < MIN_RVOL:
        return None, f"RVOL {rvol:.2f} below {MIN_RVOL}"

    if bias == "CALL":
        if prev_high is None:
            return None, "PDH unavailable — target feasibility cannot be verified"
        if target >= prev_high > price:
            return None, f"2R target blocked by PDH {prev_high:.2f}"

    if bias == "PUT":
        if prev_low is None:
            return None, "PDL unavailable — target feasibility cannot be verified"
        if target <= prev_low < price:
            return None, f"2R target blocked by PDL {prev_low:.2f}"

    return {
        "symbol": symbol,
        "bias": bias,
        "score": round(max(0, min(100, score)), 1),
        "components": components,
        "price": price,
        "trigger": float(trigger),
        "sl_underlying": float(sl),
        "target_underlying": float(target),
        "risk_points_underlying": float(risk_points),
        "risk_atr": float(risk_points / a),
        "atr": float(a),
        "atr_pct": float(a / price * 100),
        "rvol": float(rvol),
        "adx": float(adx_val),
        "rsi": float(rsi_val),
        "vwap": float(svwap),
        "ema9": float(ema9),
        "ema21": float(ema21),
        "relative_strength": float(relative),
        "market_regime": market["regime"],
        "lot_size": int(meta["lot_size"]),
        "future_token": int(meta["future_token"]),
        "nse_token": int(meta["nse_token"]),
        "signal_id": str(uuid.uuid4()),
    }, "PASS"


# ---------------------------------------------------------------------------
# OPTION SELECTION
# ---------------------------------------------------------------------------

def trading_days_to(expiry):
    today = now_ist().date()
    return int(np.busday_count(today, expiry))


def option_candidates(symbol, bias, spot):
    if nfo_df.empty:
        return []

    x = nfo_df[
        (nfo_df["name"] == symbol) &
        (nfo_df["instrument_type"].isin(["CE", "PE"]))
    ].copy()
    if x.empty:
        return []

    x["expiry"] = pd.to_datetime(x["expiry"]).dt.date
    today = now_ist().date()
    x = x[x["expiry"] >= today]
    if x.empty:
        return []

    x["dte"] = x["expiry"].apply(trading_days_to)

    # -----------------------------------------------------------------------
    # EXPIRY SELECTION
    #
    # Normal case:
    #   Prefer the nearest expiry within MIN_DTE..MAX_DTE.
    #
    # Expiry-day / no normal-window expiry:
    #   Do not use today's expiry.
    #   Fall back to the next available expiry rather than returning
    #   zero option contracts.
    # -----------------------------------------------------------------------

    typ = "CE" if bias == "CALL" else "PE"
    x = x[x["instrument_type"] == typ]

    if x.empty:
        return []

    # Never use an option that has already expired.
    x = x[x["expiry"] >= today]

    if x.empty:
        return []

    # Normal preferred expiry window.
    preferred = x[
        (x["dte"] >= MIN_DTE) &
        (x["dte"] <= MAX_DTE)
        ].copy()

    if AVOID_EXPIRY_DAY:
        preferred = preferred[preferred["dte"] >= 2]

    if not preferred.empty:
        # Use the nearest expiry inside the normal DTE window.
        expiry = preferred["expiry"].min()
        x = preferred[preferred["expiry"] == expiry].copy()

    else:
        # -------------------------------------------------------------------
        # FALLBACK:
        # No expiry exists inside the normal DTE window.
        #
        # Use the next available future expiry.
        # This prevents expiry-calendar gaps from producing zero candidates.
        # -------------------------------------------------------------------
        future = x[x["dte"] > 0].copy()

        if future.empty:
            return []

        expiry = future["expiry"].min()
        x = future[future["expiry"] == expiry].copy()


    strikes_unique = sorted(x["strike"].unique())
    if len(strikes_unique) > 2:
        step = float(np.median(np.diff(strikes_unique)))
    else:
        step = 50.0

    if bias == "CALL":
        preferred = [spot - step, spot, spot + step]
    else:
        preferred = [spot + step, spot, spot - step]

    strikes = sorted(
        strikes_unique,
        key=lambda s: min(abs(s - p) for p in preferred)
    )[:6]
    return x[x["strike"].isin(strikes)].to_dict("records")


def fetch_option_quality(rows):
    symbols = [f"NFO:{r['tradingsymbol']}" for r in rows]
    quotes = get_quote(symbols)
    out = []

    for row in rows:
        key = f"NFO:{row['tradingsymbol']}"
        q = quotes.get(key, {})
        depth = q.get("depth", {}) or {}
        buy = (depth.get("buy") or [{}])[0]
        sell = (depth.get("sell") or [{}])[0]

        bid = float(buy.get("price", 0) or 0)
        ask = float(sell.get("price", 0) or 0)
        ltp = float(q.get("last_price", 0) or 0)
        oi = int(q.get("oi", 0) or 0)
        volume = int(q.get("volume", 0) or 0)

        if bid <= 0 or ask <= 0:
            continue

        mid = (bid + ask) / 2
        spread = (ask - bid) / mid * 100 if mid > 0 else 999
        entry = ask

        # Kite does not expose a guaranteed option-delta field in every
        # quote response. We therefore use a conservative approximation
        # ONLY for risk feasibility, never for directional scoring.
        delta = q.get("delta")
        if delta is None:
            delta = 0.50
        try:
            delta = abs(float(delta))
        except Exception:
            delta = 0.50
        delta = max(0.25, min(0.90, delta))

        out.append({
            **row,
            "ltp": ltp,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread,
            "oi": oi,
            "volume": volume,
            "entry_premium": entry,
            "delta": delta,
            "premium_ok": MIN_OPTION_PREMIUM <= entry <= MAX_OPTION_PREMIUM,
        })

    return out


def select_option(setup):
    rows = option_candidates(
        setup["symbol"],
        setup["bias"],
        setup["price"],
    )

    # -----------------------------------------------------------------------
    # DIAGNOSTIC LOGGING ONLY
    #
    # IMPORTANT:
    # This section does NOT change option selection, thresholds, scoring,
    # expiry, DTE, strike selection, or trading behaviour.
    # It only records WHY option candidates are rejected.
    # -----------------------------------------------------------------------
    if not rows:
        log_event(
            "option_filter_audit",
            symbol=setup["symbol"],
            bias=setup["bias"],
            underlying_score=setup["score"],
            underlying_price=setup["price"],
            option_candidates=0,
            quality_quotes=0,
            valid_options=0,
            rejected_by_oi=0,
            rejected_by_volume=0,
            rejected_by_spread=0,
            rejected_by_premium=0,
            rejected_by_risk=0,
            rejected_by_quote=0,
            best_rejected_option=None,
            result="NO_ELIGIBLE_OPTION_CONTRACTS",
        )
        return None, "no eligible option contracts"

    qualities = fetch_option_quality(rows)

    valid = []
    reasons = []

    # Diagnostic counters ONLY.
    rejected_by_oi = 0
    rejected_by_volume = 0
    rejected_by_spread = 0
    rejected_by_premium = 0
    rejected_by_risk = 0
    rejected_by_quote = len(rows) - len(qualities)

    # Keep every rejected option with its actual market data so we can
    # identify the closest/best rejected contract later.
    rejected_options = []

    for q in qualities:
        if q["oi"] < MIN_OPTION_OI:
            rejected_by_oi += 1
            reasons.append(
                f"{q['tradingsymbol']}: OI {q['oi']} < {MIN_OPTION_OI}"
            )
            rejected_options.append({
                **q,
                "rejection_reason": "OI",
            })
            continue

        if q["volume"] < MIN_OPTION_VOLUME:
            rejected_by_volume += 1
            reasons.append(
                f"{q['tradingsymbol']}: volume {q['volume']} < {MIN_OPTION_VOLUME}"
            )
            rejected_options.append({
                **q,
                "rejection_reason": "VOLUME",
            })
            continue

        if q["spread_pct"] > MAX_OPTION_SPREAD_PCT:
            rejected_by_spread += 1
            reasons.append(
                f"{q['tradingsymbol']}: spread {q['spread_pct']:.2f}%"
            )
            rejected_options.append({
                **q,
                "rejection_reason": "SPREAD",
            })
            continue

        if not q["premium_ok"]:
            rejected_by_premium += 1
            reasons.append(
                f"{q['tradingsymbol']}: premium "
                f"{q['entry_premium']:.1f} outside range"
            )
            rejected_options.append({
                **q,
                "rejection_reason": "PREMIUM",
            })
            continue

        estimated_option_risk_points = (
            setup["risk_points_underlying"]
            * q["delta"]
            * DELTA_RISK_BUFFER
        )

        estimated_rupee_risk = (
            estimated_option_risk_points
            * setup["lot_size"]
        )

        if estimated_rupee_risk > MAX_RISK_PER_TRADE:
            rejected_by_risk += 1
            reasons.append(
                f"{q['tradingsymbol']}: "
                f"one-lot risk ₹{estimated_rupee_risk:.0f}"
            )
            rejected_options.append({
                **q,
                "rejection_reason": "RISK",
                "estimated_option_risk_points": estimated_option_risk_points,
                "estimated_rupee_risk": estimated_rupee_risk,
            })
            continue

        liquidity_score = 0
        liquidity_score += (
            4 if q["spread_pct"] <= PREFERRED_OPTION_SPREAD_PCT else 2
        )
        liquidity_score += 3 if q["oi"] >= 50_000 else 1
        liquidity_score += 3 if q["volume"] >= 10_000 else 1

        strike_score = 0
        distance = abs(float(q["strike"]) - setup["price"])

        if distance <= 0.5 * setup["atr"]:
            strike_score += 5

        if setup["bias"] == "CALL" and q["strike"] <= setup["price"]:
            strike_score += 3

        if setup["bias"] == "PUT" and q["strike"] >= setup["price"]:
            strike_score += 3

        q["estimated_option_risk_points"] = estimated_option_risk_points
        q["estimated_rupee_risk"] = estimated_rupee_risk
        q["option_score"] = liquidity_score + strike_score
        valid.append(q)

    # -----------------------------------------------------------------------
    # Find the "best rejected option" for forensic analysis.
    #
    # This ranking is ONLY for logging. It has absolutely no effect on
    # which option is selected.
    # -----------------------------------------------------------------------
    best_rejected = None

    if rejected_options:
        best_rejected = sorted(
            rejected_options,
            key=lambda q: (
                q.get("oi", 0),
                q.get("volume", 0),
                -q.get("spread_pct", 999),
            ),
            reverse=True,
        )[0]

    # -----------------------------------------------------------------------
    # DETAILED OPTION FILTER AUDIT
    # -----------------------------------------------------------------------
    log_event(
        "option_filter_audit",
        symbol=setup["symbol"],
        bias=setup["bias"],
        underlying_score=setup["score"],
        underlying_price=setup["price"],
        underlying_rvol=setup.get("rvol"),
        market_regime=setup.get("market_regime"),

        option_candidates=len(rows),
        quality_quotes=len(qualities),
        valid_options=len(valid),

        rejected_by_oi=rejected_by_oi,
        rejected_by_volume=rejected_by_volume,
        rejected_by_spread=rejected_by_spread,
        rejected_by_premium=rejected_by_premium,
        rejected_by_risk=rejected_by_risk,
        rejected_by_quote=rejected_by_quote,

        best_rejected_option=(
            {
                "tradingsymbol": best_rejected.get("tradingsymbol"),
                "strike": best_rejected.get("strike"),
                "expiry": best_rejected.get("expiry"),
                "dte": best_rejected.get("dte"),
                "oi": best_rejected.get("oi"),
                "volume": best_rejected.get("volume"),
                "bid": best_rejected.get("bid"),
                "ask": best_rejected.get("ask"),
                "ltp": best_rejected.get("ltp"),
                "spread_pct": best_rejected.get("spread_pct"),
                "entry_premium": best_rejected.get("entry_premium"),
                "rejection_reason": best_rejected.get("rejection_reason"),
            }
            if best_rejected
            else None
        ),

        result="PASS" if valid else "REJECTED",
    )

    if not valid:
        return (
            None,
            "; ".join(reasons[:5])
            or "no option passed liquidity/risk filters"
        )

    valid.sort(
        key=lambda x: (
            x["option_score"],
            -x["spread_pct"],
            x["oi"],
        ),
        reverse=True,
    )

    return valid[0], "PASS"


# ---------------------------------------------------------------------------
# CANDIDATE SCANNER
# ---------------------------------------------------------------------------

def scan_candidates():
    global last_candidate_refresh, candidate_cache

    if time.time() - last_candidate_refresh < CANDIDATE_REFRESH_SECONDS:
        return list(candidate_cache.values())

    reset_day_if_needed()
    if now_ist().time() < ENTRY_START:
        return []

    market = get_market_context()
    current_state["market_regime"] = market["regime"]
    current_state["market_score"] = market["score"]
    current_state["nifty"] = market["nifty"]
    current_state["vix"] = market["vix"]

    if market["regime"] in ("UNKNOWN", "STALE_DATA"):
        current_state["reason"] = "Nifty 5m data is not from today — skipping scan"
        return []

    underlying_setups = []
    rejections = []

    active_symbols = list(stock_universe.keys())

    # Stage 1: rank the underlying setups. No option-chain calls here.
    # This keeps the API load bounded even with ~40 active stocks.
    for symbol in active_symbols:
        if ONE_TRADE_PER_STOCK_PER_DAY and symbol in traded_stocks_today:
            rejections.append({
                "symbol": symbol,
                "stage": "trade_limit",
                "reason": "already traded today",
            })
            continue

        try:
            setup, reason = detect_setup(symbol, stock_universe[symbol], market)
            if not setup:
                rejections.append({
                    "symbol": symbol,
                    "stage": "setup",
                    "reason": reason,
                })
                continue
            underlying_setups.append(setup)
        except Exception as exc:
            logging.exception("Underlying candidate error %s", symbol)
            rejections.append({
                "symbol": symbol,
                "stage": "exception",
                "reason": type(exc).__name__,
            })

    underlying_setups.sort(key=lambda x: x["score"], reverse=True)

    # Stage 2: option quality is expensive. Only inspect the strongest
    # underlying setups. The final ranking then includes option execution.
    option_checked = 0
    results = []
    for setup in underlying_setups[:8]:
        option_checked += 1
        try:
            option, option_reason = select_option(setup)
            if not option:
                rejections.append({
                    "symbol": setup["symbol"],
                    "stage": "option",
                    "reason": option_reason,
                })
                continue

            # Keep the stock's directional score independent from
            # option execution quality.
            #
            # The stock score answers:
            # "How good is the underlying trade?"
            #
            # The option score answers:
            # "How good is the option vehicle?"

            setup["alpha_score"] = round(setup["score"], 1)
            setup["execution_score"] = option["option_score"]

            setup["option"] = option
            setup["option_reason"] = option_reason
            results.append(setup)

        except Exception as exc:
            logging.exception("Option candidate error %s", setup["symbol"])
            rejections.append({
                "symbol": setup["symbol"],
                "stage": "option_exception",
                "reason": type(exc).__name__,
            })

    # Underlying setups below the top 8 are intentionally recorded so we can
    # later determine whether the option pre-filter was too aggressive.
    for setup in underlying_setups[8:]:
        rejections.append({
            "symbol": setup["symbol"],
            "stage": "option_deferred",
            "reason": "underlying rank below top-8 option inspection cutoff",
        })

    results.sort(key=lambda x: x["alpha_score"], reverse=True)
    candidate_cache = {x["symbol"]: x for x in results}
    last_candidate_refresh = time.time()

    current_state["candidate_count"] = len(results)
    current_state["top_candidates"] = [
        candidate_summary(x) for x in results[:8]
    ]
    current_state["rejections"] = rejections[-100:]

    for x in results:
        append_csv(
            os.path.join(LOG_DIR, "stock_candidate_log.csv"),
            candidate_log_row(x, selected=False),
        )
    for r in rejections:
        append_csv(
            os.path.join(LOG_DIR, "stock_candidate_rejections.csv"),
            {
                "timestamp": now_ist(),
                "symbol": r["symbol"],
                "stage": r["stage"],
                "reason": r["reason"],
            },
        )

    log_event(
        "candidate_scan",
        market_regime=market["regime"],
        underlying_setups=len(underlying_setups),
        option_checked=option_checked,
        candidates=len(results),
        rejections=len(rejections),
    )
    return results


def candidate_summary(x):
    o = x.get("option", {})
    return {
        "rank": None,
        "symbol": x["symbol"],
        "bias": x["bias"],
        "score": x["score"],
        "price": x["price"],
        "rvol": x["rvol"],
        "adx": x["adx"],
        "rsi": x["rsi"],
        "relative_strength": x["relative_strength"],
        "sl": x["sl_underlying"],
        "target": x["target_underlying"],
        "risk_points": x["risk_points_underlying"],
        "risk_atr": x["risk_atr"],
        "option": o.get("tradingsymbol"),
        "strike": o.get("strike"),
        "expiry": o.get("expiry"),
        "dte": o.get("dte"),
        "option_entry": o.get("entry_premium"),
        "spread_pct": o.get("spread_pct"),
        "oi": o.get("oi"),
        "option_volume": o.get("volume"),
        "estimated_rupee_risk": o.get("estimated_rupee_risk"),
    }


def candidate_log_row(x, selected=False):
    o = x.get("option", {})
    return {
        "timestamp": now_ist(),
        "signal_id": x["signal_id"],
        "symbol": x["symbol"],
        "bias": x["bias"],
        "score": x["score"],
        "selected": selected,
        # Individual score components — preserved for post-trade analysis.
        "score_trend_structure": x.get("components", {}).get("trend_structure", 0),
        "score_breakout_quality": x.get("components", {}).get("breakout_quality", 0),
        "score_rvol": x.get("components", {}).get("rvol", 0),
        "score_relative_strength": x.get("components", {}).get("relative_strength", 0),
        "score_momentum": x.get("components", {}).get("momentum", 0),
        "score_market_alignment": x.get("components", {}).get("market_alignment", 0),
        "price": x["price"],
        "trigger": x["trigger"],
        "sl_underlying": x["sl_underlying"],
        "target_underlying": x["target_underlying"],
        "risk_points": x["risk_points_underlying"],
        "risk_atr": x["risk_atr"],
        "atr": x["atr"],
        "rvol": x["rvol"],
        "adx": x["adx"],
        "rsi": x["rsi"],

        # Preserve raw RS value, but explicitly mark invalid values.
        "relative_strength": (
            float(x["relative_strength"])
            if x.get("relative_strength") is not None
               and math.isfinite(float(x["relative_strength"]))
            else None
        ),
        "relative_strength_valid": (
                x.get("relative_strength") is not None
                and math.isfinite(float(x["relative_strength"]))
        ),
        "market_regime": x["market_regime"],
        "option_symbol": o.get("tradingsymbol"),
        "option_strike": o.get("strike"),
        "expiry": o.get("expiry"),
        "dte": o.get("dte"),
        "option_entry": o.get("entry_premium"),
        "option_bid": o.get("bid"),
        "option_ask": o.get("ask"),
        "option_spread_pct": o.get("spread_pct"),
        "option_oi": o.get("oi"),
        "option_volume": o.get("volume"),
        "estimated_rupee_risk": o.get("estimated_rupee_risk"),
    }


# ---------------------------------------------------------------------------
# SHADOW TRADING
# ---------------------------------------------------------------------------

def choose_trade(candidates):
    global active_trade, daily_trades, daily_risk_deployed

    if active_trade is not None:
        return None, "trade already active"
    if daily_trades >= MAX_TRADES_PER_DAY:
        return None, "daily trade limit reached"
    if daily_risk_deployed >= MAX_DAILY_RISK:
        return None, f"daily risk cap reached (₹{daily_risk_deployed:.0f} >= ₹{MAX_DAILY_RISK:.0f})"

    if not candidates:
        return None, "no qualified candidates"

    top = candidates[0]
    if top["alpha_score"] < MIN_RANK_SCORE:
        return None, f"top score {top['alpha_score']:.1f} < {MIN_RANK_SCORE}"

    # Version 1 is deliberately conservative: one top trade is selected.
    # The second-trade allowance is kept as a later research switch.
    selected = top
    pending_pullbacks.pop(selected["symbol"], None)

    option = selected["option"]
    now = now_ist()

    q = get_quote([f"NSE:{selected['symbol']}", f"NFO:{option['tradingsymbol']}"])
    uq = q.get(f"NSE:{selected['symbol']}", {})
    oq = q.get(f"NFO:{option['tradingsymbol']}", {})

    underlying = float(uq.get("last_price", selected["price"]) or selected["price"])



    depth = oq.get("depth", {}) or {}

    ask = float((depth.get("sell") or [{}])[0].get("price", 0) or 0)
    bid = float((depth.get("buy") or [{}])[0].get("price", 0) or 0)
    if ask <= 0 or bid <= 0:
        return None, "entry option market disappeared"

    spread = (ask - bid) / ((ask + bid) / 2) * 100
    if spread > MAX_OPTION_SPREAD_PCT:
        return None, f"entry spread widened to {spread:.2f}%"

    # V2: anti-chase check is measured from EMA9,
    # not from a breakout trigger.
    ema9 = float(selected.get("ema9", underlying) or underlying)
    atr_at_entry = float(selected["atr"])

    if selected["bias"] == "CALL":
        extension = max(0, underlying - ema9) / atr_at_entry
    else:
        extension = max(0, ema9 - underlying) / atr_at_entry

    if extension > MAX_ENTRY_EXTENSION_ATR:
        return None, (
            f"entry moved {extension:.2f} ATR beyond EMA9"
        )

    entry_premium = ask
    estimated_risk = option["estimated_rupee_risk"]

    signal_id = selected["signal_id"]

    # --- Option SL risk cap ---
    calc_sl = max(0.05, entry_premium - option["estimated_option_risk_points"])
    floor_sl = entry_premium * 0.85
    final_sl = max(calc_sl, floor_sl)

    trade = {
        "signal_id": signal_id,
        "symbol": selected["symbol"],
        "bias": selected["bias"],
        "option_symbol": option["tradingsymbol"],
        "strike": float(option["strike"]),
        "expiry": str(option["expiry"]),
        "dte": int(option["dte"]),
        "lot_size": int(selected["lot_size"]),
        "entry_time": now.isoformat(),
        "entry_underlying": underlying,
        "trigger": float(selected["trigger"]),
        "atr_at_entry": float(selected["atr"]),
        "entry_premium": entry_premium,
        "breakout_failure_polls": 0,
        "bid_at_entry": bid,
        "ask_at_entry": ask,
        "spread_pct": spread,

        # Underlying structural levels — actual exit logic continues to use these.
        "underlying_sl": selected["sl_underlying"],
        "underlying_target": selected["target_underlying"],
        "risk_points_underlying": selected["risk_points_underlying"],
        "option_risk_points": option["estimated_option_risk_points"],
        "option_sl": final_sl,
        "initial_option_sl": final_sl,
        "current_option_sl": final_sl,
        "option_target": (entry_premium + option["estimated_option_risk_points"] * TARGET_R_MULTIPLE),
        "profit_lock_stage": 0,
        "target_r_multiple": TARGET_R_MULTIPLE,
        "estimated_rupee_risk": estimated_risk,
        "mfe_underlying": underlying,
        "mae_underlying": underlying,
        "mfe_option": entry_premium,
        "mae_option": entry_premium,
        "score": selected["score"],
        "rvol": selected["rvol"],
        "adx": selected["adx"],
        "rsi": selected["rsi"],

        # Raw RS value for post-trade analysis.
        "relative_strength": (
            float(selected["relative_strength"])
            if selected.get("relative_strength") is not None
               and math.isfinite(float(selected["relative_strength"]))
            else None
        ),
        "relative_strength_valid": (
                selected.get("relative_strength") is not None
                and math.isfinite(float(selected["relative_strength"]))
        ),
        "market_regime": selected["market_regime"],
        "option_oi": option["oi"],
        "option_volume": option["volume"],
        "vix_at_entry": current_state.get("vix", {}).get("ltp"),
        "status": "ACTIVE_SHADOW",
        "exit_reason": None,
    }

    active_trade = trade
    daily_trades += 1
    daily_risk_deployed += estimated_risk
    traded_stocks_today.add(selected["symbol"])

    append_csv(
        os.path.join(LOG_DIR, "stock_candidate_log.csv"),
        candidate_log_row(selected, selected=True),
    )
    append_csv(
        os.path.join(LOG_DIR, "stock_shadow_trades.csv"),
        {
            "timestamp": now,
            "event": "ENTRY",
            **trade,
        },
    )
    log_event("shadow_entry", **trade)
    save_state()
    return trade, "SHADOW ENTRY"


def monitor_active_trade():
    if active_trade is None:
        return

    t = active_trade
    key = f"NFO:{t['option_symbol']}"
    ukey = f"NSE:{t['symbol']}"

    q = get_quote([key, ukey])
    oq = q.get(key, {})
    uq = q.get(ukey, {})

    option_ltp = float(oq.get("last_price", 0) or 0)

    depth = oq.get("depth", {}) or {}

    option_bid = float(
        (depth.get("buy") or [{}])[0].get("price", 0) or 0
    )

    option_ask = float(
        (depth.get("sell") or [{}])[0].get("price", 0) or 0
    )

    # For a long option:
    # entry = ASK
    # exit  = BID
    executable_exit = option_bid if option_bid > 0 else option_ltp


    underlying = float(uq.get("last_price", 0) or 0)

    if option_ltp <= 0 or underlying <= 0:
        return

    # Underlying is useful for logging/excursion tracking,
    # but it must NOT block option-premium exit monitoring.
    if underlying <= 0:
        underlying = t.get("current_underlying", t["entry_underlying"])

    t["current_option"] = executable_exit
    t["current_underlying"] = underlying

    # Option premium excursion is direction-agnostic (a long option's
    # favorable direction is always "up" regardless of underlying bias).
    t["mfe_option"] = max(t.get("mfe_option", t["entry_premium"]), option_ltp)
    t["mae_option"] = min(t.get("mae_option", t["entry_premium"]), option_ltp)

    # Underlying excursion IS direction-dependent. For a PUT, favorable
    # movement is downward, so "MFE" must track the lowest price seen,
    # not the highest. The old code used max()/min() unconditionally,
    # which silently inverted MFE/MAE for every PUT trade.
    if t["bias"] == "CALL":
        t["mfe_underlying"] = max(t.get("mfe_underlying", t["entry_underlying"]), underlying)
        t["mae_underlying"] = min(t.get("mae_underlying", t["entry_underlying"]), underlying)
    else:
        t["mfe_underlying"] = min(t.get("mfe_underlying", t["entry_underlying"]), underlying)
        t["mae_underlying"] = max(t.get("mae_underlying", t["entry_underlying"]), underlying)

    exit_reason = None



    # ---------------------------------------------------------
    # DYNAMIC OPTION-PREMIUM PROFIT PROTECTION
    # ---------------------------------------------------------
    entry_premium = float(t["entry_premium"])
    risk_points = float(t["option_risk_points"])

    if risk_points > 0:
        current_r = (option_ltp - entry_premium) / risk_points

        # Stage 1: once trade reaches +1R,
        # protect +0.25R.
        if current_r >= PROFIT_LOCK_1_R:
            lock_price = (entry_premium + risk_points * PROFIT_LOCK_1_LOCK_R)

            if lock_price > t["current_option_sl"]:
                t["current_option_sl"] = lock_price
                t["profit_lock_stage"] = max(t.get("profit_lock_stage", 0),1,)

        # Stage 2: once trade reaches +1.5R,
        # protect +0.75R.
        if current_r >= PROFIT_LOCK_2_R:
            lock_price = (entry_premium + risk_points * PROFIT_LOCK_2_LOCK_R)

            if lock_price > t["current_option_sl"]:
                t["current_option_sl"] = lock_price
                t["profit_lock_stage"] = 2

        t["current_r"] = current_r

    # ---------------------------------------------------------
    # EXIT CHECKS
    # ---------------------------------------------------------

    # UNDERLYING STRUCTURAL SL — CATASTROPHIC BACKSTOP ONLY
    underlying_sl = float(t.get("underlying_sl", 0) or 0)
    underlying_sl_hit = False

    if underlying_sl > 0:
        if t["bias"] == "CALL" and underlying <= underlying_sl:
            underlying_sl_hit = True
        elif t["bias"] == "PUT" and underlying >= underlying_sl:
            underlying_sl_hit = True

    if underlying_sl_hit:
        exit_reason = "UNDERLYING_STRUCTURAL_SL"

    # PRIMARY OPTION-PREMIUM EXIT LOGIC
    elif option_ltp <= t["current_option_sl"]:
        exit_reason = (
            "OPTION_PROFIT_LOCK"
            if t.get("profit_lock_stage", 0) > 0
            else "OPTION_SL"
        )

    elif option_ltp >= t["option_target"]:
        exit_reason = "OPTION_2R_TARGET"


    # Mandatory end-of-day exit remains unchanged.
    if now_ist().time() >= HARD_EXIT:
        exit_reason = "3:15_HARD_EXIT"

    if exit_reason:
        exit_trade(option_ltp, underlying, exit_reason)


def exit_trade(option_price, underlying_price, reason):
    global active_trade, daily_pnl

    if active_trade is None:
        return

    t = active_trade
    pnl_points = float(option_price - t["entry_premium"])
    pnl_rupees = pnl_points * t["lot_size"]
    daily_pnl += pnl_rupees

    t.update({
        "exit_time": now_ist().isoformat(),
        "exit_option": option_price,
        "exit_underlying": underlying_price,
        "pnl_points": pnl_points,
        "pnl_rupees": pnl_rupees,
        "exit_reason": reason,
        "status": "CLOSED_SHADOW",
    })

    append_csv(
        os.path.join(LOG_DIR, "stock_shadow_trades.csv"),
        {
            "timestamp": now_ist(),
            "event": "EXIT",
            **t,
        },
    )
    log_event("shadow_exit", **t)

    active_trade = None
    save_state()


# ---------------------------------------------------------------------------
# MAIN ENGINE LOOP
# ---------------------------------------------------------------------------

def engine_tick():
    global current_state

    reset_day_if_needed()

    now = now_ist()
    current_state["last_scan"] = now.strftime("%H:%M:%S")
    current_state["daily_trades"] = daily_trades
    current_state["daily_pnl"] = round(daily_pnl, 2)
    current_state["daily_risk_deployed"] = round(daily_risk_deployed, 2)
    current_state["active_trade"] = active_trade
    current_state["shadow_mode"] = True

    if now.time() < SCAN_START:
        current_state["decision"] = "WAIT"
        current_state["reason"] = f"Waiting for scan start {SCAN_START.strftime('%H:%M')}"
        emit_state()
        return

    if now.time() >= HARD_EXIT:
        if active_trade:
            monitor_active_trade()
        current_state["decision"] = "MARKET CLOSED"
        current_state["reason"] = "3:15 PM hard exit / end of session"
        current_state["active_trade"] = active_trade
        emit_state(force=True)
        return

    if active_trade is not None:
        monitor_active_trade()
        current_state["decision"] = "SHADOW TRADE ACTIVE"
        current_state["reason"] = (
            f"{active_trade['symbol']} {active_trade['bias']} — fixed 2R target"
        )
        current_state["active_trade"] = active_trade
        emit_state()
        return

    if now.time() > ENTRY_CUTOFF:
        current_state["decision"] = "NO TRADE"
        current_state["reason"] = "Entry cutoff reached"
        emit_state()
        return

    try:
        refresh_universe()
        if not stock_universe:
            current_state["decision"] = "NO TRADE"
            current_state["reason"] = "No qualified F&O universe"
            emit_state()
            return

        candidates = scan_candidates()
        trade, reason = choose_trade(candidates)

        if trade:
            current_state["decision"] = "SHADOW ENTRY"
            current_state["reason"] = (
                f"{trade['symbol']} {trade['bias']} | "
                f"Score {trade['score']:.1f} | "
                f"Fixed 1:{TARGET_R_MULTIPLE:.0f}"
            )
        else:
            current_state["decision"] = "NO TRADE"
            current_state["reason"] = reason

        current_state["active_trade"] = active_trade
        current_state["daily_trades"] = daily_trades
        current_state["daily_pnl"] = round(daily_pnl, 2)
        current_state["daily_risk_deployed"] = round(daily_risk_deployed, 2)
        emit_state()
    except Exception as exc:
        logging.exception("Engine tick failed")
        current_state["decision"] = "NO TRADE"
        current_state["reason"] = f"Engine error: {type(exc).__name__}"
        emit_state(force=True)


def engine_loop():
    logging.info("Stock Options Ranker started | %s", STRATEGY_VERSION)
    while True:
        try:
            reset_day_if_needed()
            if market_hours():
                engine_tick()
            else:
                current_state["decision"] = "WAIT"
                current_state["reason"] = "Outside market hours"
                emit_state()
        except Exception:
            logging.exception("Engine loop error")
        time.sleep(
            ACTIVE_TRADE_REFRESH_SECONDS
            if active_trade is not None
            else CANDIDATE_REFRESH_SECONDS
        )


# ---------------------------------------------------------------------------
# WEB
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("templates", "orderflow_stock_options.html")


@app.route("/health")
def health():
    return {
        "ok": True,
        "strategy": STRATEGY_VERSION,
        "shadow_mode": True,
        "time": now_ist().isoformat(),
    }


@socketio.on("connect")
def on_connect(auth=None):
    emit_state(force=True)


# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "stock_options_orderflow.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    load_state()
    reset_day_if_needed()

    load_nfo_instruments()
    build_fno_universe()

    thread = threading.Thread(target=engine_loop, daemon=True)
    thread.start()

    print("=" * 70)
    print("STOCK OPTIONS RANKER — SHADOW MODE")
    print(f"Version: {STRATEGY_VERSION}")
    print(f"Port: {PORT}")
    print(f"Universe max: {MAX_UNIVERSE_SIZE}")
    print(f"Risk/trade: ₹{MAX_RISK_PER_TRADE:.0f}")
    print(f"Target: 1:{TARGET_R_MULTIPLE:.0f}")
    print("LIVE ORDER PLACEMENT: DISABLED")
    print("=" * 70)

    socketio.run(
        app,
        host="0.0.0.0",
        port=PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
