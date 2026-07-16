# === CRUDE OIL ORDER-FLOW BUYER ENGINE v2.8 (SHADOW MODE) ===
# v2.8:
#   - Full dashboard fields added (dte, spot, scenario, adx, momentum, etc.)
#   - Emits signal on every scan, even NO TRADE, to keep frontend updated
#   - All previous fixes retained (exit-checks unconditional, caching, etc.)

import json
import csv
import time
import datetime
import logging
import threading
import os
import calendar
import pytz
import uuid
from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import math
from scipy.stats import norm, pearsonr
import concurrent.futures


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


load_dotenv(".env.orderflow")

# === CONFIG ===
API_KEY = os.getenv("OF_API_KEY")
ACCESS_TOKEN = os.getenv("OF_ACCESS_TOKEN")
if not API_KEY or not ACCESS_TOKEN:
    print("❌ OF_API_KEY or OF_ACCESS_TOKEN not found in .env.orderflow")
    exit(1)

PORT = 8064
MAX_LOTS = 2
CRUDE_LOT_SIZE = 100
STATE_FILE = "crude_orderflow_state.json"
CRUDE_TRAIL_ACTIVATION = 15
CRUDE_BREAKEVEN_PCT = 0.12   # lock breakeven once profit hits 12% of entry premium
CRUDE_TRAIL_FLOOR = 15       # tightest the trail can ratchet down to
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

STRATEGY_VERSION = "v2.8"
ENTRY_COOLDOWN_SECONDS = 300
MAX_SPREAD_PCT = 10.0
HTF_MISMATCH_PENALTY = 15   # points deducted when 1H VWAP disagrees with entry bias

VOLATILITY_THRESHOLD_HIGH = 1.5
VOLATILITY_THRESHOLD_MODERATE = 0.8

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "crude_orderflow.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

IST = pytz.timezone('Asia/Kolkata')
def now_ist():
    return datetime.datetime.now(IST).replace(tzinfo=None)

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# === GLOBALS ===
state_lock = threading.RLock()
current_signal = {"decision": "NO TRADE", "reason": "Initializing..."}
trade_entry_time = None
entry_option_ltp = None
active_trade = None
last_exit_time = None
daily_pnl = 0
max_daily_loss = -4500
daily_reset_date = now_ist().date()

_candle_cache = {}
cached_candles = pd.DataFrame()
cached_ht_candles = pd.DataFrame()
cached_day_candles = pd.DataFrame()
last_candles_time = None
last_ht_candles_time = None
last_day_candles_time = None
mcx_instruments_cache = None
mcx_instruments_cache_time = None
expiry_cache = None
last_cache_time = None

# ======================== TIMEOUT WRAPPER ========================
def kite_call_with_timeout(func, *args, timeout=5, **kwargs):
    executor = None
    future = None
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logging.warning(f"⏰ TIMEOUT: {func.__name__} (>{timeout}s)")
        if future:
            future.cancel()
        return None
    except Exception as e:
        logging.warning(f"❌ ERROR in {func.__name__}: {e}")
        return None
    finally:
        if executor:
            # Shutdown without waiting for hung threads to avoid freezing
            executor.shutdown(wait=False)

# ======================== JSON LOGGING ========================
def _json_safe_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)  # last-resort fallback so a write NEVER silently fails

def log_json(event_type, data):
    json_log_path = os.path.join(LOG_DIR, "crude_orderflow.jsonl")
    try:
        with open(json_log_path, 'a') as f:
            f.write(json.dumps({
                "timestamp": now_ist().isoformat(),
                "event": event_type,
                **data
            }, default=_json_safe_default) + "\n")   # ✅ added default
    except Exception as e:
        logging.warning(f"JSON log write failed: {e}")

# ======================== SAFE CSV APPEND (schema-drift proof) ========================
def append_csv_row_safe(csv_path, row):
    """
    Append `row` to csv_path. If the file doesn't exist yet, writes a fresh header.
    If it exists but `row` has different keys than the file's current header
    (e.g. a new field was added after the baseline started), migrates the whole
    file to a unified header — backfilling old rows with blanks for new columns —
    instead of silently writing misaligned columns.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, 'r', newline='') as f:
        existing_header = next(csv.reader(f), [])

    if list(row.keys()) == existing_header:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=existing_header)
            writer.writerow(row)
        return

    new_fields = [k for k in row.keys() if k not in existing_header]
    logging.warning(f"⚠️ CSV schema changed for {csv_path} — migrating file to add columns: {new_fields}")
    df_old = pd.read_csv(csv_path)
    unified_fields = existing_header + new_fields
    for col in unified_fields:
        if col not in df_old.columns:
            df_old[col] = ""
    df_old = df_old[unified_fields]
    df_new = pd.DataFrame([{k: row.get(k, "") for k in unified_fields}])
    pd.concat([df_old, df_new], ignore_index=True).to_csv(csv_path, index=False)

# ======================== FEATURE FUNCTIONS ========================
def compute_rvol(candles, period=5):
    if len(candles) < period + 1:
        return {"value": 1.0, "score": 0, "reason": "insufficient data"}
    avg_vol = candles['volume'].iloc[-period-1:-1].mean()
    if avg_vol == 0:
        return {"value": 1.0, "score": 0, "reason": "zero avg volume"}
    rvol = candles['volume'].iloc[-1] / avg_vol
    # Continuous scoring: starts contributing from RVOL 0.3, caps at 20
    score = min(20, max(0, (rvol - 0.3) * 30))
    return {"value": rvol, "score": round(score, 2), "reason": f"RVOL {rvol:.2f}"}

def compute_oi_acceleration(candles, period=5):
    if 'oi' not in candles.columns or len(candles) < period + 2:
        return {"value": 0, "score": 0, "reason": "insufficient OI data"}
    oi_vals = candles['oi'].iloc[-period-1:].values
    if len(oi_vals) < period:
        return {"value": 0, "score": 0, "reason": "not enough OI"}
    diffs = np.diff(oi_vals)
    if len(diffs) < 2:
        return {"value": 0, "score": 0, "reason": "not enough diff"}
    accel = diffs[-1] - diffs[-2]
    score = min(18, max(-10, accel / 1000 * 2))
    return {"value": accel, "score": round(score, 2), "reason": f"OI accel {accel:.0f}"}

def compute_trend_efficiency(candles, period=10):
    if len(candles) < period + 1:
        return {"value": 0, "score": 0, "reason": "insufficient data"}
    close = candles['close'].iloc[-period-1:]
    net = close.iloc[-1] - close.iloc[0]
    abs_sum = np.abs(close.diff()).sum()
    if abs_sum == 0:
        return {"value": 0, "score": 0, "reason": "no movement"}
    eff = net / abs_sum
    score = min(15, max(0, (eff + 0.3) * 25))
    return {"value": eff, "score": round(score, 2), "reason": f"Efficiency {eff:.2f}"}

def compute_relative_range(candles, period=10):
    if len(candles) < period + 1:
        return {"value": 1.0, "score": 0, "reason": "insufficient data"}
    current_range = candles['high'].iloc[-1] - candles['low'].iloc[-1]
    avg_range = (candles['high'].iloc[-period-1:-1] - candles['low'].iloc[-period-1:-1]).mean()
    if avg_range == 0:
        return {"value": 1.0, "score": 0, "reason": "zero avg range"}
    rel = current_range / avg_range
    # Continuous scoring: starts contributing from 0.5, caps at 15
    score = min(15, max(0, (rel - 0.5) * 25))
    return {"value": rel, "score": round(score, 2), "reason": f"RelRange {rel:.2f}"}

def compute_breakout_acceptance(candles, key_levels):
    if candles.empty or len(candles) < 2:
        return {"value": 0, "score": 0, "reason": "no data"}
    pdh = key_levels.get("PDH")
    pdl = key_levels.get("PDL")
    if pdh is None or pdl is None:
        return {"value": 0, "score": 0, "reason": "no PDH/PDL"}
    close = candles['close'].iloc[-1]
    high = candles['high'].iloc[-1]
    low = candles['low'].iloc[-1]
    if high > pdh and close > pdh * 0.998:
        return {"value": 1, "score": 6, "reason": "Breakout above PDH accepted"}
    elif low < pdl and close < pdl * 1.002:
        return {"value": 1, "score": 6, "reason": "Breakout below PDL accepted"}
    return {"value": 0, "score": 0, "reason": "No breakout"}

def compute_market_regime(candles, atr, vwap):
    if candles.empty:
        return "NEUTRAL"
    price = candles['close'].iloc[-1]
    atr_pct = (atr / price) * 100 if price > 0 else 0

    if atr_pct > VOLATILITY_THRESHOLD_HIGH:
        regime = "HIGH_VOLATILITY"
    elif atr_pct > VOLATILITY_THRESHOLD_MODERATE:
        regime = "MODERATE_VOLATILITY"
    else:
        regime = "LOW_VOLATILITY"

    if vwap and price > vwap:
        regime += "_BULLISH"
    elif vwap and price < vwap:
        regime += "_BEARISH"
    return regime

def compute_interaction_bonus(feature_scores):
    bonus = 0
    rvol = feature_scores.get("rvol", {}).get("value", 1)
    te = feature_scores.get("trend_efficiency", {}).get("value", 0)
    breakout = feature_scores.get("breakout_acceptance", {}).get("value", 0)
    oi_acc = feature_scores.get("oi_acceleration", {}).get("value", 0)
    rel_range = feature_scores.get("relative_range", {}).get("value", 1)

    if rvol > 1.5 and te > 0.3:
        bonus += 5
    if breakout == 1 and oi_acc > 0:
        bonus += 5
    if rel_range > 1.3 and te > 0.2:
        bonus += 3
    return bonus

def compute_scenario_probs(base_score, bias):
    if bias == "CALL":
        upside = 33.3 + (base_score - 50) * 0.5
        downside = 33.3 - (base_score - 50) * 0.5
    elif bias == "PUT":
        downside = 33.3 + (base_score - 50) * 0.5
        upside = 33.3 - (base_score - 50) * 0.5
    else:
        upside = downside = 33.3
    flat = 100 - upside - downside
    upside = max(0, min(100, upside))
    downside = max(0, min(100, downside))
    flat = max(0, min(100, flat))
    return {"upside": round(upside, 1), "downside": round(downside, 1), "flat": round(flat, 1)}

# ======================== SCORE DISTRIBUTION LOGGER ========================
_score_lock = threading.Lock()

def log_score_distribution(now, total_score, base_score, bonus, interaction_bonus, bias,
                           futures_ltp, market_regime, dte, reason):
    """Append score details to logs/score_distribution.csv on every scan."""
    score_log_path = os.path.join(LOG_DIR, "score_distribution.csv")
    try:
        with _score_lock:
            write_header = not os.path.exists(score_log_path)
            with open(score_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "total_score", "base_score", "bonus", "interaction_bonus",
                        "bias", "futures_ltp", "market_regime", "dte", "reason"
                    ])
                writer.writerow([
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    round(total_score, 2),
                    round(base_score, 2),
                    round(bonus, 2),
                    round(interaction_bonus, 2),
                    bias,
                    round(futures_ltp, 2),
                    market_regime,
                    dte,
                    reason
                ])
    except Exception as e:
        logging.warning(f"Score distribution log failed: {e}")


# ======================== COMPOSITE SCORE ========================
def composite_score(candles, price_chg, oi_chg, key_levels):
    reasons = []
    score = 0
    bias = "NEUTRAL"
    vwap = key_levels.get("VWAP")
    # FIX: Ensure vwap is not None AND greater than 0 to prevent false CALL bias on 0 volume/VWAP mismatche
    if vwap is not None and vwap > 0:
        price = candles['close'].iloc[-1]
        if price > vwap:
            score += 20
            bias = "CALL" if bias == "NEUTRAL" else bias
            reasons.append("Price above VWAP")
        else:
            score += 10
            bias = "PUT" if bias == "NEUTRAL" else bias
            reasons.append("Price below VWAP")
    oi_class = "NEUTRAL"
    if oi_chg != 0:
        if price_chg > 0 and oi_chg > 0:
            oi_class = "FRESH_LONGS"
            score += 20
            bias = "CALL" if bias != "PUT" else bias
            reasons.append("Fresh longs building")
        elif price_chg > 0 and oi_chg < 0:
            oi_class = "SHORT_COVERING"
            score += 15
            bias = "CALL" if bias != "PUT" else bias
            reasons.append("Short covering")
        elif price_chg < 0 and oi_chg > 0:
            oi_class = "FRESH_SHORTS"
            score += 20
            bias = "PUT" if bias != "CALL" else bias
            reasons.append("Fresh shorts building")
        elif price_chg < 0 and oi_chg < 0:
            oi_class = "LONG_UNWINDING"
            score += 15
            bias = "PUT" if bias != "CALL" else bias
            reasons.append("Long unwinding")
    if not candles.empty and len(candles) >= 20:
        bins = pd.cut(candles['close'], bins=20, precision=1)
        vol_by_bin = candles.groupby(bins, observed=False)['volume'].sum()
        poc_bin = vol_by_bin.idxmax()
        poc = poc_bin.mid if hasattr(poc_bin, 'mid') else None
        if poc is not None:
            price = candles['close'].iloc[-1]
            if price > poc * 0.995:
                score += 15
                reasons.append("Price near POC")
            else:
                score += 5
    if len(candles) >= 5:
        high = candles['high'].iloc[-1]
        close = candles['close'].iloc[-1]
        prev_high = candles['high'].iloc[-2]
        avg_vol = candles['volume'].iloc[-5:].mean()
        if high > prev_high and candles['volume'].iloc[-1] > avg_vol * 1.2 and close < prev_high:
            score -= 10
            reasons.append("Bull trap detected")
    if len(candles) >= 10:
        swing_high = candles['high'].iloc[-10:-1].max()
        swing_low = candles['low'].iloc[-10:-1].min()
        if candles['high'].iloc[-1] > swing_high and candles['close'].iloc[-1] < swing_high:
            score += 10
            reasons.append("Stop hunt above")
        if candles['low'].iloc[-1] < swing_low and candles['close'].iloc[-1] > swing_low:
            score += 10
            reasons.append("Stop hunt below")
    if len(candles) >= 14:
        high, low, close = candles['high'], candles['low'], candles['close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        atr_safe = atr if atr != 0 else 1
        dm_plus = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
        dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)
        di_plus = 100 * dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr_safe
        di_minus = 100 * dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr_safe
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1)
        adx = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss_safe = loss if loss != 0 else 1
        rs = gain / loss_safe
        rsi = 100 - (100 / (1 + rs))
        if adx > 25 and ((bias == "CALL" and rsi > 50) or (bias == "PUT" and rsi < 50)):
            score += 10
            reasons.append("Trend momentum confirms bias")
        elif adx < 20:
            score -= 5
            reasons.append("Weak trend – cautious")
    score = max(0, min(100, score))
    if bias == "NEUTRAL":
        bias = "CALL" if score > 50 else "PUT" if score > 40 else "NEUTRAL"
    return {"score": score, "bias": bias, "oi_class": oi_class, "reasons": reasons}

# ======================== CRUDE HELPERS ========================
def resolve_current_expiry():
    global expiry_cache, last_cache_time
    try:
        now = now_ist()
        if expiry_cache and last_cache_time and (now - last_cache_time).total_seconds() < 300:
            return expiry_cache
        instruments = get_mcx_instruments(force=False)
        futs = [i for i in instruments if i.get('name') == 'CRUDEOIL' and i.get('instrument_type') == 'FUT']
        futs.sort(key=lambda x: x['expiry'])
        active = []
        for f in futs:
            exp = f['expiry']
            if isinstance(exp, datetime.date):
                exp_date = exp
            else:
                exp_date = datetime.datetime.strptime(str(exp), '%Y-%m-%d').date()
            if exp_date > now_ist().date():
                active.append(f)
        expiry_cache = active[0] if active else None
        last_cache_time = now
        return expiry_cache
    except Exception as e:
        logging.error(f"Expiry resolution error: {e}")
        return expiry_cache

def get_mcx_instruments(force=False):
    global mcx_instruments_cache, mcx_instruments_cache_time
    now = now_ist()
    if not force and mcx_instruments_cache and mcx_instruments_cache_time and (now - mcx_instruments_cache_time).total_seconds() < 86400:
        return mcx_instruments_cache
    try:
        mcx_instruments_cache = kite_call_with_timeout(kite.instruments, "MCX")
        if mcx_instruments_cache is None:
            mcx_instruments_cache = []
        mcx_instruments_cache_time = now
        logging.info(f"MCX instrument cache refreshed: {len(mcx_instruments_cache)} instruments")
        return mcx_instruments_cache
    except Exception as e:
        logging.error(f"MCX instruments fetch error: {e}")
        return mcx_instruments_cache or []

def get_candles(token, interval="15minute", days=5, force=False):
    global cached_candles, last_candles_time, cached_day_candles, last_day_candles_time
    now = now_ist()

    if interval == "15minute" and not force and last_candles_time and (now - last_candles_time).total_seconds() < 90:
        return cached_candles

    if interval == "day" and not force and last_day_candles_time and (now - last_day_candles_time).total_seconds() < 300:
        return cached_day_candles

    try:
        to_date = now
        from_date = to_date - datetime.timedelta(days=days)
        data = kite_call_with_timeout(kite.historical_data, token, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"), interval, oi=(interval == "15minute"))
        if data is None:
            data = []
        df = pd.DataFrame(data)
        if interval == "15minute":
            cached_candles = df
            last_candles_time = now
        elif interval == "day":
            cached_day_candles = df
            last_day_candles_time = now
        return df
    except Exception as e:
        logging.warning(f"Historical data error: {e}")
        if interval == "15minute":
            return cached_candles if not cached_candles.empty else pd.DataFrame()
        elif interval == "day":
            return cached_day_candles if not cached_day_candles.empty else pd.DataFrame()
        else:
            return pd.DataFrame()

def get_higher_tf_candles(token, force=False):
    global cached_ht_candles, last_ht_candles_time
    now = now_ist()
    if not force and last_ht_candles_time and (now - last_ht_candles_time).total_seconds() < 180:
        return cached_ht_candles
    try:
        to_date = now
        from_date = to_date - datetime.timedelta(days=10)
        data = kite_call_with_timeout(kite.historical_data, token, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"), "60minute")
        if data is None:
            data = []
        df = pd.DataFrame(data)
        cached_ht_candles = df
        last_ht_candles_time = now
        return df
    except Exception as e:
        logging.warning(f"HTF data error: {e}")
        return cached_ht_candles if not cached_ht_candles.empty else pd.DataFrame()

def resolve_crude_fut_token():
    expiry_data = resolve_current_expiry()
    if not expiry_data:
        return None, None
    return expiry_data['instrument_token'], expiry_data['tradingsymbol']

def calculate_vwap(df):
    if df.empty or 'volume' not in df.columns or df['volume'].sum() == 0:
        return df['close'].iloc[-1] if not df.empty else 0
    df = df.copy()
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    return (df['typical'] * df['volume']).sum() / df['volume'].sum()

def get_key_levels(df_daily, df_intraday):
    pdh = df_daily['high'].iloc[-2] if len(df_daily) >= 2 else None
    pdl = df_daily['low'].iloc[-2] if len(df_daily) >= 2 else None

    if not df_intraday.empty:
        today = now_ist().date()
        df_today = df_intraday[df_intraday['date'].dt.date == today] if 'date' in df_intraday.columns else df_intraday
        if not df_today.empty:
            typical = (df_today['high'] + df_today['low'] + df_today['close']) / 3
            vwap = (typical * df_today['volume']).cumsum() / df_today['volume'].cumsum()
            if pd.isna(vwap.iloc[-1]):
                vwap_val = None
            else:
                vwap_val = round(vwap.iloc[-1], 2)
        else:
            vwap_val = None
    else:
        vwap_val = None
    return {"PDH": pdh, "PDL": pdl, "VWAP": vwap_val}

def safe_emit(event, data):
    print(f"🔥 safe_emit: {event}")
    try:
        socketio.emit(event, data)
        logging.info(f"EMIT {event}: {data.get('decision')} | {data.get('reason')}")
    except Exception as e:
        logging.exception(f"safe_emit FAILED for {event}: {e}")
        print(f"❌ safe_emit FAILED: {e}")

# ======================== FORCE CLOSE ========================
def force_close_trade(reason_tag, log_prefix="FORCE CLOSE", underlying_ltp=None, is_sim=False):
    global trade_entry_time, entry_option_ltp, active_trade, daily_pnl, last_exit_time
    if trade_entry_time is None or active_trade is None or entry_option_ltp is None:
        trade_entry_time = entry_option_ltp = active_trade = None
        save_state()
        return 0

    exit_ltp = active_trade.get('option_ltp', entry_option_ltp)
    if active_trade.get('symbol'):
        try:
            q = kite_call_with_timeout(kite.quote, [f"MCX:{active_trade['symbol']}"])
            if q is None:
                q = {}
            depth = q.get(f"MCX:{active_trade['symbol']}", {}).get('depth', {})
            fresh_bid = depth.get('buy', [{}])[0].get('price', 0)
            if fresh_bid > 0:
                exit_ltp = fresh_bid  # simulate a realistic sell fill at the bid
            else:
                fresh = q.get(f"MCX:{active_trade['symbol']}", {}).get('last_price')
                if fresh and fresh > 0:
                    exit_ltp = fresh
                    logging.warning(f"⚠️ No bid price for {active_trade.get('symbol')}, falling back to LTP")
        except Exception as e:
            logging.warning(
                f"⚠️ Fresh exit-price fetch failed for {active_trade.get('symbol')} — using last known price {exit_ltp} for PnL calc: {e}")

    lots = active_trade.get('lots', 1)
    entry = entry_option_ltp
    exit_pnl = (exit_ltp - entry) * lots * CRUDE_LOT_SIZE
    daily_pnl += exit_pnl

    highest = active_trade.get('highest_premium', entry)
    lowest = active_trade.get('lowest_premium', entry)
    mfe_pts = (highest - entry) * lots * CRUDE_LOT_SIZE
    mae_pts = max(0, (entry - lowest) * lots * CRUDE_LOT_SIZE)

    sl_price = active_trade.get('sl_price', entry - 38)
    risk_per_lot = abs(entry - sl_price) * CRUDE_LOT_SIZE
    r_multiple = exit_pnl / risk_per_lot if risk_per_lot != 0 else 0

    prefix = "[SIM] " if is_sim else ""
    logging.info(f"{prefix}{log_prefix} — {reason_tag} — PnL: ₹{exit_pnl:.0f} | Daily: ₹{daily_pnl:.0f} | R: {r_multiple:.2f}")

    log_json("TRADE_CLOSED", {
        "signal_id": active_trade.get('signal_id'),
        "strategy_version": STRATEGY_VERSION,
        "entry_time": trade_entry_time.isoformat() if trade_entry_time else "",
        "exit_time": now_ist().isoformat(),
        "bias": active_trade.get('bias', ''),
        "strike": active_trade.get('strike', ''),
        "symbol": active_trade.get('symbol', ''),
        "entry_price": entry,
        "exit_price": exit_ltp,
        "underlying_entry": active_trade.get('underlying_ltp', 0),
        "underlying_exit": underlying_ltp if underlying_ltp else 0,
        "pnl": exit_pnl,
        "r_multiple": round(r_multiple, 2),
        "mfe_pts": round(mfe_pts, 2),
        "mae_pts": round(mae_pts, 2),
        "holding_minutes": round((now_ist() - trade_entry_time).total_seconds() / 60, 1),
        "exit_reason": reason_tag,
        "daily_pnl": daily_pnl,
        "market_regime": active_trade.get('market_regime', ''),
        "signal_quality": active_trade.get('signal_quality', 0),
        "breakeven_locked": active_trade.get('breakeven_locked', False),
        "trail_activated": active_trade.get('trail_active', False),
        "entry_spread_pct": active_trade.get('entry_spread_pct', 0),
        "adx": active_trade.get('adx', 0),
        "rsi": active_trade.get('rsi', 0),
        "entry_atr": active_trade.get('entry_atr', 0),
        "vix_value": active_trade.get('vix_value', 0),
        "is_sim": is_sim
    })

    try:
        regime = active_trade.get('market_regime', 'UNKNOWN')
        regime_file = os.path.join(LOG_DIR, "crude_regime_performance.csv")
        regime_exists = os.path.exists(regime_file)
        if regime_exists:
            df_reg = pd.read_csv(regime_file)
        else:
            df_reg = pd.DataFrame(columns=["regime", "trades", "wins", "win_rate", "avg_pnl"])
        if regime in df_reg['regime'].values:
            idx = df_reg[df_reg['regime'] == regime].index[0]
            df_reg.loc[idx, 'trades'] += 1
            if exit_pnl > 0:
                df_reg.loc[idx, 'wins'] += 1
            df_reg.loc[idx, 'win_rate'] = df_reg.loc[idx, 'wins'] / df_reg.loc[idx, 'trades']
            avg_pnl = (df_reg.loc[idx, 'avg_pnl'] * (df_reg.loc[idx, 'trades'] - 1) + exit_pnl) / df_reg.loc[idx, 'trades']
            df_reg.loc[idx, 'avg_pnl'] = avg_pnl
        else:
            new_row = {"regime": regime, "trades": 1, "wins": 1 if exit_pnl > 0 else 0,
                       "win_rate": 1 if exit_pnl > 0 else 0, "avg_pnl": exit_pnl}
            df_reg = pd.concat([df_reg, pd.DataFrame([new_row])], ignore_index=True)
        df_reg.to_csv(regime_file, index=False)
    except Exception as e:
        logging.warning(f"Regime stats update failed: {e}")

    trade_entry_time = entry_option_ltp = active_trade = None
    last_exit_time = now_ist()
    save_state()
    return exit_pnl

# ======================== SAVE/LOAD STATE ========================
def save_state():
    state = {
        "trade_entry_time": trade_entry_time.isoformat() if trade_entry_time else None,
        "entry_option_ltp": entry_option_ltp,
        "active_trade": active_trade,
        "daily_pnl": daily_pnl,
        "last_exit_time": last_exit_time.isoformat() if last_exit_time else None,
        "daily_reset_date": daily_reset_date.isoformat() if daily_reset_date else None
    }
    try:
        tmp_path = STATE_FILE + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, default=_json_safe_default)
        os.replace(tmp_path, STATE_FILE)  # atomic — never leaves a truncated file on disk
    except Exception as e:
        logging.error(f"🚨 save_state FAILED — active_trade may not persist across restart: {e}")

def load_state():
    global trade_entry_time, entry_option_ltp, active_trade, daily_pnl, daily_reset_date, last_exit_time
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        if state.get("trade_entry_time"):
            trade_entry_time = datetime.datetime.fromisoformat(state["trade_entry_time"])
        entry_option_ltp = state.get("entry_option_ltp")
        active_trade = state.get("active_trade")
        # --- NEW: convert last_quote_time ---
        if active_trade and active_trade.get("last_quote_time"):
            try:
                active_trade["last_quote_time"] = datetime.datetime.fromisoformat(active_trade["last_quote_time"])
            except (TypeError, ValueError):
                active_trade["last_quote_time"] = now_ist()
        daily_pnl = state.get("daily_pnl", 0)
        if state.get("daily_reset_date"):
            daily_reset_date = datetime.date.fromisoformat(state["daily_reset_date"])
        if state.get("last_exit_time"):
            last_exit_time = datetime.datetime.fromisoformat(state["last_exit_time"])
        else:
            last_exit_time = None
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logging.error(f"🚨 load_state FAILED — starting with clean state (state file may be corrupted): {e}")
        trade_entry_time = entry_option_ltp = active_trade = None
        daily_pnl = 0
        daily_reset_date = now_ist().date()
        last_exit_time = None
        return False

# ======================== MAIN SCAN ========================
def run_crude_orderflow_scan():
    global current_signal, trade_entry_time, entry_option_ltp, active_trade, daily_pnl, daily_reset_date, cached_candles
    print(f"🔍 Crude scan running at {now_ist().strftime('%H:%M:%S')}")
    with state_lock:
        try:
            now = now_ist()
            if daily_reset_date is None:
                daily_reset_date = now_ist().date()

            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=23, minute=30, second=0, microsecond=0)
            if not (market_open <= now <= market_close):
                if active_trade:
                    force_close_trade("Market Closed", "MARKET CLOSE", None, is_sim=True)
                # Emit a NO TRADE signal so the dashboard updates
                current_signal = {"decision": "NO TRADE", "reason": "Market Closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                return

            if now_ist().date() != daily_reset_date:
                daily_pnl = 0
                daily_reset_date = now_ist().date()

            # === STEP 1: EXIT CHECKS (UNCONDITIONAL) ===
            if active_trade is not None:
                opt_symbol = active_trade.get('symbol')
                if opt_symbol:
                    # We'll track whether we got a fresh valid price
                    fresh_price_received = False
                    stale_duration = (now - active_trade.get('last_quote_time', now)).total_seconds()
                    try:
                        opt_quote = kite_call_with_timeout(kite.quote, [f"MCX:{opt_symbol}"])
                        if opt_quote is not None:
                            depth = opt_quote.get(f"MCX:{opt_symbol}", {}).get('depth', {})
                            ltp = opt_quote.get(f"MCX:{opt_symbol}", {}).get('last_price', 0)
                            bid = depth.get('buy', [{}])[0].get('price', 0)

                            # Use bid for trailing, fallback to LTP
                            if bid > 0:
                                trail_premium = bid
                                fresh_price_received = True
                            elif ltp > 0:
                                trail_premium = ltp
                                fresh_price_received = True
                                logging.warning(f"⚠️ No bid for {opt_symbol}, using LTP for trail")
                            else:
                                trail_premium = 0
                                logging.warning(f"⚠️ Zero price for {opt_symbol} – treating as stale")

                            if fresh_price_received:
                                # Update stored price and timestamp
                                active_trade['option_ltp'] = ltp
                                active_trade['trail_premium'] = trail_premium
                                active_trade['last_quote_time'] = now
                            else:
                                # No fresh price – check staleness
                                if stale_duration > 10:
                                    logging.warning(f"🚨 Stale quote for {opt_symbol} for >10s – forcing exit.")
                                    underlying_price = active_trade.get('underlying_ltp', 0)
                                    exit_pnl = force_close_trade("STALE QUOTE TIMEOUT", "STALE QUOTE", underlying_price,
                                                                 is_sim=True)
                                    current_signal = {"decision": "EXIT — STALE QUOTE",
                                                      "reason": f"Quote stale >10s | PnL: ₹{exit_pnl:.0f}"}
                                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                                    safe_emit('crude_orderflow_signal', current_signal)
                                    return
                        else:
                            # Quote fetch failed (timeout or None) – also treat as stale
                            logging.warning(
                                f"⚠️ Quote fetch failed for {opt_symbol} (stale for {stale_duration:.1f}s) – "
                                f"using stale price {active_trade.get('option_ltp', entry_option_ltp)}."
                            )
                            if stale_duration > 10:
                                logging.warning(f"🚨 Stale quote for {opt_symbol} for >10s – forcing exit.")
                                underlying_price = active_trade.get('underlying_ltp', 0)
                                exit_pnl = force_close_trade("STALE QUOTE TIMEOUT", "STALE QUOTE", underlying_price,
                                                             is_sim=True)
                                current_signal = {"decision": "EXIT — STALE QUOTE",
                                                  "reason": f"Quote stale >10s | PnL: ₹{exit_pnl:.0f}"}
                                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                                safe_emit('crude_orderflow_signal', current_signal)
                                return
                    except Exception as e:
                        logging.warning(
                            f"⚠️ CRUDE option premium refresh exception for {opt_symbol}: {e} – treating as stale"
                        )
                        # Also check staleness here
                        if stale_duration > 10:
                            logging.warning(f"🚨 Stale quote for {opt_symbol} for >10s (exception) – forcing exit.")
                            underlying_price = active_trade.get('underlying_ltp', 0)
                            exit_pnl = force_close_trade("STALE QUOTE TIMEOUT", "STALE QUOTE", underlying_price,
                                                         is_sim=True)
                            current_signal = {"decision": "EXIT — STALE QUOTE",
                                              "reason": f"Quote stale >10s (exception) | PnL: ₹{exit_pnl:.0f}"}
                            current_signal["last_scan"] = now.strftime("%H:%M:%S")
                            safe_emit('crude_orderflow_signal', current_signal)
                            return

                # --- Use trail_premium for high-water mark and trailing stop ---
                current_premium = active_trade.get('trail_premium', entry_option_ltp)
                highest_premium = active_trade.get('highest_premium', entry_option_ltp)
                lowest_premium = active_trade.get('lowest_premium', entry_option_ltp)

                if current_premium > highest_premium:
                    active_trade['highest_premium'] = current_premium
                    highest_premium = current_premium
                if current_premium < lowest_premium:
                    active_trade['lowest_premium'] = current_premium

                # --- BREAKEVEN LOCK: once MFE clears CRUDE_BREAKEVEN_PCT of entry premium, SL -> entry ---
                if not active_trade.get('breakeven_locked', False):
                    breakeven_trigger = entry_option_ltp * CRUDE_BREAKEVEN_PCT
                    if highest_premium - entry_option_ltp >= breakeven_trigger:
                        active_trade['sl_price'] = max(active_trade.get('sl_price', entry_option_ltp - 38),
                                                       entry_option_ltp)
                        active_trade['breakeven_locked'] = True
                        logging.info(
                            f"🔒 [SIM] Breakeven lock engaged | entry {entry_option_ltp} | peak {highest_premium}")

                underlying_ltp = active_trade.get('underlying_ltp', 0)
                fut_sym = active_trade.get('fut_sym')
                if fut_sym:
                    try:
                        q = kite_call_with_timeout(kite.quote, [f"MCX:{fut_sym}"])
                        if q:
                            underlying_ltp = q.get(f"MCX:{fut_sym}", {}).get('last_price', underlying_ltp)
                    except Exception as e:
                        logging.warning(
                            f"⚠️ Underlying LTP refresh failed for {fut_sym} (using stale price {underlying_ltp}): {e}")

                sl_price = active_trade.get('sl_price', max(entry_option_ltp - 38, 10.0))
                active_trade['sl_price'] = sl_price
                if current_premium <= sl_price:
                    exit_pnl = force_close_trade(f"SL HIT (₹{abs(entry_option_ltp - sl_price):.0f})", "STOP LOSS",
                                                 underlying_ltp, is_sim=True)
                    current_signal = {"decision": "EXIT — STOP LOSS", "reason": f"SL hit | PnL: ₹{exit_pnl:.0f}"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('crude_orderflow_signal', current_signal)
                    return

                # --- PROFIT-RATCHETING TRAIL: tighten trail distance as MFE grows, floor at CRUDE_TRAIL_FLOOR ---
                base_trail = active_trade.get('trail_distance', 20)
                mfe = highest_premium - entry_option_ltp
                if mfe >= base_trail * 3:
                    trail_distance = max(CRUDE_TRAIL_FLOOR, base_trail * 0.5)
                elif mfe >= base_trail * 2:
                    trail_distance = max(CRUDE_TRAIL_FLOOR, base_trail * 0.7)
                else:
                    trail_distance = base_trail

                active_trade['trail_distance'] = trail_distance
                activation_threshold = active_trade.get('activation_threshold',
                                                        CRUDE_TRAIL_ACTIVATION)  # CRUDE_TRAIL_ACTIVATION in nifty
                if not active_trade.get('trail_active', False):
                    if current_premium >= entry_option_ltp + activation_threshold:
                        active_trade['trail_active'] = True

                if active_trade.get('trail_active', False):
                    if highest_premium - current_premium >= trail_distance:
                        exit_pnl = force_close_trade(
                            f"TRAILING STOP (peak {round(highest_premium, 2)}, exit {round(current_premium, 2)})",
                            "TRAILING STOP", underlying_ltp, is_sim=True)
                        current_signal = {"decision": "EXIT — TRAILING STOP",
                                          "reason": f"Peak {round(highest_premium, 2)} → {round(current_premium, 2)} | PnL: ₹{exit_pnl:.0f}"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('crude_orderflow_signal', current_signal)  # 'crude_orderflow_signal' in  crude
                        return


            # === COOLDOWN: wait after an exit before re-entering ===
            if active_trade is None and last_exit_time:
                elapsed = (now - last_exit_time).total_seconds()
                if elapsed < ENTRY_COOLDOWN_SECONDS:
                    remaining = int(ENTRY_COOLDOWN_SECONDS - elapsed)
                    current_signal = {"decision": "NO TRADE", "reason": f"Cooldown ({remaining}s remaining)"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('crude_orderflow_signal', current_signal)
                    # print(f"🔴 REJECTED: Cooldown ({remaining}s remaining)")
                    return

            # === STEP 2: ENTRY SIGNAL COMPUTATION ===
            if daily_pnl <= max_daily_loss:
                # print(f"🔴 REJECTED: Daily loss cap reached (PnL: {daily_pnl})")
                current_signal = {"decision": "NO TRADE", "reason": f"Daily loss cap reached"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                if active_trade:
                    monitor_signal = {
                        "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                        "symbol": active_trade.get('symbol', ''),
                        "is_monitoring": True,
                        "strike": active_trade.get('strike'),
                        "entry_ltp": round(entry_option_ltp, 2),
                        "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                        "option_sl": round(active_trade.get('sl_price', max(entry_option_ltp - 38, 10.0)), 2),
                        "lots": active_trade.get('lots', 1),
                        "setup_quality": active_trade.get('setup_quality', 0),
                        "signal_quality": active_trade.get('signal_quality', 0),
                        "market_regime": active_trade.get('market_regime', ''),
                        "signal_id": active_trade.get('signal_id'),
                        "last_scan": now.strftime("%H:%M:%S"),
                        "primary_reason": "[SIM] Monitoring only (loss cap reached)",
                        "highest_premium": round(active_trade.get('highest_premium', entry_option_ltp), 2),
                        "trail_stop": round(
                            active_trade.get('highest_premium', entry_option_ltp) - active_trade.get('trail_distance',
                                                                                                     0),
                            2) if active_trade.get('trail_active', False) else None,
                        "trail_active": active_trade.get('trail_active', False),
                        "max_loss": round((entry_option_ltp - active_trade.get('sl_price', max(entry_option_ltp - 38,
                                                                                               10.0))) * CRUDE_LOT_SIZE, 0),
                    }
                    safe_emit('crude_orderflow_signal', monitor_signal)
                return

            expiry_data = resolve_current_expiry()
            if not expiry_data:
                # print("🔴 REJECTED: Expiry resolution failed")
                current_signal = {"decision": "NO TRADE", "reason": "Expiry resolution failed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                return
            fut_token = expiry_data['instrument_token']
            fut_sym = expiry_data['tradingsymbol']

            instruments_all = get_mcx_instruments()
            opt_expiries = sorted(set(
                i['expiry'] for i in instruments_all
                if i.get('name') == 'CRUDEOIL' and i.get('instrument_type') in ('CE', 'PE')
                and i['expiry'] >= now_ist().date()
            ))

            # --- Rollover: if DTE == 0, use the next available expiry ---
            if opt_expiries:
                nearest_option_expiry = opt_expiries[0]
                dte = (nearest_option_expiry - now_ist().date()).days
                # If today is expiry day, roll over to the next expiry (if available)
                if dte == 0 and len(opt_expiries) > 1:
                    nearest_option_expiry = opt_expiries[1]
                    dte = (nearest_option_expiry - now_ist().date()).days
                    logging.info(f"🔄 Rolled over to next expiry: {nearest_option_expiry} (DTE={dte})")
            else:
                nearest_option_expiry = None
                dte = (expiry_data['expiry'] - now_ist().date()).days if expiry_data else 0

            expiry_date_str = nearest_option_expiry.isoformat() if nearest_option_expiry else None


            # --- Expiry-Day Cutoff (no new entries near expiry) ---
            if active_trade is None:
                cutoff = None
                if dte == 0:
                    cutoff = now.replace(hour=21, minute=0, second=0, microsecond=0)  # 9 PM on expiry day
                elif dte == 1:
                    cutoff = now.replace(hour=22, minute=0, second=0, microsecond=0)  # 10 PM day before expiry
                if cutoff and now >= cutoff:
                    current_signal = {"decision": "NO TRADE",
                                      "reason": f"No entries on DTE {dte} after {cutoff.strftime('%H:%M')}"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('crude_orderflow_signal', current_signal)
                    # print(f"🔴 REJECTED: DTE {dte} cutoff passed")
                    return

            try:
                quote = kite_call_with_timeout(kite.quote, [f"MCX:{fut_sym}"])
                if quote is None:
                    quote = {}
                futures_ltp = quote.get(f"MCX:{fut_sym}", {}).get('last_price', 0)
                if futures_ltp <= 0:
                    current_signal = {"decision": "NO TRADE", "reason": "Failed to fetch futures LTP"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('crude_orderflow_signal', current_signal)
                    return
            except Exception as e:
                logging.warning(f"⚠️ Futures quote fetch exception for {fut_sym}: {e}")
                current_signal = {"decision": "NO TRADE",
                                  "reason": f"Exception fetching futures quote: {type(e).__name__}"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                return


            candles_15m = get_candles(fut_token, "15minute", 5)
            candles_day = get_candles(fut_token, "day", 10)

            entry_atr = 0
            if not candles_15m.empty and len(candles_15m) >= 14:
                tr = pd.concat([
                    candles_15m['high'] - candles_15m['low'],
                    (candles_15m['high'] - candles_15m['close'].shift()).abs(),
                    (candles_15m['low'] - candles_15m['close'].shift()).abs()
                ], axis=1).max(axis=1)
                entry_atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

            vwap = calculate_vwap(candles_15m.tail(40)) if not candles_15m.empty else futures_ltp
            key_levels = {"VWAP": vwap, "PDH": candles_day['high'].iloc[-2] if len(candles_day)>=2 else None,
                          "PDL": candles_day['low'].iloc[-2] if len(candles_day)>=2 else None}

            rvol = compute_rvol(candles_15m)
            oi_acc = compute_oi_acceleration(candles_15m)
            trend_eff = compute_trend_efficiency(candles_15m)
            rel_range = compute_relative_range(candles_15m)
            breakout = compute_breakout_acceptance(candles_15m, key_levels)

            feature_scores = {
                "rvol": rvol,
                "oi_acceleration": oi_acc,
                "trend_efficiency": trend_eff,
                "relative_range": rel_range,
                "breakout_acceptance": breakout
            }

            price_chg = candles_15m['close'].iloc[-1] - candles_15m['close'].iloc[-2] if len(candles_15m) >= 2 else 0
            oi_chg = 0
            if 'oi' in candles_15m.columns and len(candles_15m) >= 2:
                raw_oi_chg = candles_15m['oi'].iloc[-1] - candles_15m['oi'].iloc[-2]
                prev_oi = candles_15m['oi'].iloc[-2]
                # Only use OI change if it's more than 2% of previous OI (noise filter)
                if prev_oi > 0 and abs(raw_oi_chg) / prev_oi > 0.005:
                    oi_chg = raw_oi_chg
                else:
                    oi_chg = 0

            comp = composite_score(candles_15m, price_chg, oi_chg, key_levels)
            base_score = comp["score"]
            bias = comp["bias"]

            bonus = 0
            interaction_bonus = compute_interaction_bonus(feature_scores)
            total_score = base_score + bonus + interaction_bonus

            # print(
                # f"SCORE BREAKDOWN | base={base_score} bonus={bonus} interaction={interaction_bonus} total={total_score} bias={bias}")
            # print(f"FEATURES: {feature_scores}")
            # print(f"REASONS: {comp['reasons']}")

            # ADX computation for dashboard
            adx_val = 0
            rsi_val = 50
            if len(candles_15m) >= 14:
                high, low, close = candles_15m['high'], candles_15m['low'], candles_15m['close']
                tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(
                    axis=1)
                atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
                atr_safe = atr if atr != 0 else 1
                dm_plus = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(
                    lower=0)
                dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(
                    lower=0)
                di_plus = 100 * dm_plus.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
                di_minus = 100 * dm_minus.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
                dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1)
                adx_val = dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

                delta = close.diff()
                gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
                rs = 999999 if loss == 0 else gain / loss
                rsi_val = 100 - (100 / (1 + rs))


            market_regime = compute_market_regime(candles_15m, entry_atr, vwap)
            signal_quality = min(100, total_score)

            # --- Log score distribution (only when score > 40 to keep file small) ---
            if total_score > 40:
                log_score_distribution(
                    now, total_score, base_score, bonus, interaction_bonus, bias,
                    futures_ltp, market_regime, dte,
                    "Accepted" if total_score >= 57 and bias != "NEUTRAL" else "Rejected"
                )

            if total_score < 57 or bias == "NEUTRAL":
                print(f"🔴 REJECTED: Entry Score {round(total_score, 1)} < 57, Bias: {bias}")
                current_signal = {"decision": "NO TRADE", "reason": f"Entry Score {round(total_score, 1)}"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                current_signal["score_breakdown"] = {
                    "total": round(total_score, 1),
                    "threshold": 57,
                    "features": {
                        name: {
                            "score": round(v.get("score", 0), 2),
                            "reason": v.get("reason", "")
                        } for name, v in feature_scores.items()
                    },
                    "reasons": comp["reasons"]
                }
                current_signal["dte"] = dte
                current_signal["expiry_date"] = expiry_date_str
                safe_emit('crude_orderflow_signal', current_signal)

                if active_trade:
                    monitor_signal = {
                        "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                        "is_monitoring": True,
                        "symbol": active_trade.get('symbol', ''),
                        "strike": active_trade.get('strike'),
                        "entry_ltp": round(entry_option_ltp, 2),
                        "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                        "option_sl": round(active_trade.get('sl_price', max(entry_option_ltp - 38, 10.0)), 2),
                        "lots": active_trade.get('lots', 1),
                        "setup_quality": active_trade.get('setup_quality', 0),
                        "signal_quality": active_trade.get('signal_quality', 0),
                        "market_regime": active_trade.get('market_regime', ''),
                        "signal_id": active_trade.get('signal_id'),
                        "last_scan": now.strftime("%H:%M:%S"),
                        "primary_reason": f"[SIM] Monitoring (entry score {round(total_score, 1)})",
                        # Full dashboard fields
                        "dte": active_trade.get('dte', 0),
                        "vix_value": 0.0,
                        "spot_price": round(futures_ltp, 2),
                        "scenario": compute_scenario_probs(base_score, bias),
                        "vix_regime": "N/A",
                        "vix_direction": "N/A",
                        "adx": round(adx_val, 2),
                        "momentum_checks": 2 if adx_val > 25 else 1 if adx_val > 20 else 0,
                        "kills": [],
                        "failed_criteria": [],
                        "time_elapsed": round((now - trade_entry_time).total_seconds() / 60,
                                              1) if trade_entry_time else 0,
                        "highest_premium": round(active_trade.get('highest_premium', entry_option_ltp), 2),
                        "trail_stop": round(
                            active_trade.get('highest_premium', entry_option_ltp) - active_trade.get('trail_distance',
                                                                                                     0), 2)
                        if active_trade.get('trail_active', False) else None,
                        "trail_active": active_trade.get('trail_active', False),
                        "max_loss": round(
                            (entry_option_ltp - active_trade.get('sl_price',
                                                                 max(entry_option_ltp - 38, 10.0))) * CRUDE_LOT_SIZE,
                            0),
                    }
                    safe_emit('crude_orderflow_signal', monitor_signal)
                return

            ht_candles = get_higher_tf_candles(fut_token)
            if ht_candles.empty or len(ht_candles) < 8:
                current_signal = {"decision": "NO TRADE", "reason": "HTF data unavailable — failing closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                return

            ht_vwap = calculate_vwap(ht_candles.tail(20))
            if ht_vwap <= 0:
                current_signal = {"decision": "NO TRADE", "reason": "HTF VWAP invalid — failing closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                return

            ht_bias = "CALL" if ht_candles['close'].iloc[-1] > ht_vwap else "PUT"
            if ht_bias != bias:
                original_score = round(total_score, 1)
                total_score -= HTF_MISMATCH_PENALTY
                signal_quality = max(0, min(100, total_score))  # recompute — the earlier value is now stale
                logging.info(
                    f"⚠️ REJECTED: HTF mismatch ({ht_bias} vs {bias}). Penalising score {original_score} -> {round(total_score, 1)}")

                try:
                    htf_log_file = os.path.join(LOG_DIR, "crude_htf_rejections.csv")
                    append_csv_row_safe(htf_log_file, {
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "futures_price": round(futures_ltp, 2),
                        "entry_tf_bias": bias,
                        "original_score": original_score,
                        "penalised_score": round(total_score, 1),
                        "entry_tf_score": round(total_score, 1),
                        "htf_bias": ht_bias,
                        "htf_vwap": round(ht_vwap, 2),
                        "market_regime": market_regime,
                        "outcome": "PROCEEDED" if total_score >= 57 and bias != "NEUTRAL" else "STILL_REJECTED",
                    })
                except Exception as e:
                    logging.warning(f"⚠️ Failed to log HTF rejection: {e}")

                # --- REQUIRED: Crude's score threshold check runs BEFORE this block (unlike
                # Nifty's, which runs after). Without this second check, the penalty would
                # never actually filter anything — the trade already passed that gate earlier. ---
                if total_score < 57 or bias == "NEUTRAL":
                    current_signal = {"decision": "NO TRADE",
                                      "reason": f"HTF penalty dropped score below threshold ({round(total_score, 1)})"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    current_signal["dte"] = dte
                    current_signal["expiry_date"] = expiry_date_str
                    safe_emit('crude_orderflow_signal', current_signal)
                    if active_trade:
                        monitor_signal = {
                            "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                            "is_monitoring": True,
                            "symbol": active_trade.get('symbol', ''),
                            "strike": active_trade.get('strike'),
                            "entry_ltp": round(entry_option_ltp, 2),
                            "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                            "option_sl": round(active_trade.get('sl_price', max(entry_option_ltp - 38, 10.0)), 2),
                            "lots": active_trade.get('lots', 1),
                            "setup_quality": active_trade.get('setup_quality', 0),
                            "signal_quality": active_trade.get('signal_quality', 0),
                            "market_regime": active_trade.get('market_regime', ''),
                            "signal_id": active_trade.get('signal_id'),
                            "last_scan": now.strftime("%H:%M:%S"),
                            "primary_reason": "[SIM] Monitoring (HTF penalty)",
                            "highest_premium": round(active_trade.get('highest_premium', entry_option_ltp), 2),
                            "trail_stop": round(
                                active_trade.get('highest_premium', entry_option_ltp) - active_trade.get(
                                    'trail_distance', 0), 2) if active_trade.get('trail_active', False) else None,
                            "trail_active": active_trade.get('trail_active', False),
                            "max_loss": round((entry_option_ltp - active_trade.get('sl_price',
                                                                                   max(entry_option_ltp - 38,
                                                                                       10.0))) * CRUDE_LOT_SIZE, 0),
                        }
                        safe_emit('crude_orderflow_signal', monitor_signal)
                    return

            # --- LOG: distance to opposing PDH/PDL (no gating) ---
            distance_to_level_atr = None
            if bias == "CALL" and key_levels.get("PDH") and futures_ltp < key_levels["PDH"] and entry_atr > 0:
                distance_to_level_atr = round((key_levels["PDH"] - futures_ltp) / entry_atr, 2)
            elif bias == "PUT" and key_levels.get("PDL") and futures_ltp > key_levels["PDL"] and entry_atr > 0:
                distance_to_level_atr = round((futures_ltp - key_levels["PDL"]) / entry_atr, 2)


            interval = 50
            atm = round(futures_ltp / interval) * interval

            candidate_strike = atm if (bias == "CALL" and futures_ltp >= atm - 50) or (bias == "PUT" and futures_ltp <= atm + 50) else (atm + interval if bias == "CALL" else atm - interval)

            try:
                instruments = get_mcx_instruments(force=False)
                opt_matches = [i for i in instruments
                               if i.get('name') == 'CRUDEOIL'
                               and i.get('instrument_type') == ('CE' if bias == "CALL" else 'PE')
                               and i.get('strike') == candidate_strike
                               and i.get('expiry') == nearest_option_expiry]
                if opt_matches:
                    option_symbol = opt_matches[0]['tradingsymbol']
                else:
                    expiry_str = nearest_option_expiry.strftime('%d%b').upper().lstrip('0')
                    option_symbol = f"CRUDEOIL{expiry_str}{candidate_strike}{'CE' if bias == 'CALL' else 'PE'}"
            except Exception as e:
                logging.warning(
                    f"⚠️ Option symbol resolution failed | bias={bias} strike={candidate_strike} expiry={nearest_option_expiry}: {e}")
                current_signal = {"decision": "NO TRADE",
                                  "reason": f"Option symbol resolution failed: {type(e).__name__}"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('crude_orderflow_signal', current_signal)
                return

            if active_trade is None:
                try:
                    opt_quote = kite_call_with_timeout(kite.quote, [f"MCX:{option_symbol}"])
                    if opt_quote is None:
                        # print(f"🔴 REJECTED: Option quote fetch failed for {option_symbol}")
                        current_signal = {"decision": "NO TRADE", "reason": "Option quote fetch failed"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('crude_orderflow_signal', current_signal)
                        return

                    option_ltp = opt_quote.get(f"MCX:{option_symbol}", {}).get('last_price', 0)
                    # --- Spread/Liquidity Filter — moved ahead of the premium floor so the
                    # floor check uses the real fillable price (ask), not stale LTP ---
                    depth = opt_quote.get(f"MCX:{option_symbol}", {}).get('depth', {})
                    bid = depth.get('buy', [{}])[0].get('price', 0)
                    ask = depth.get('sell', [{}])[0].get('price', 0)

                    if bid <= 0 or ask <= 0:
                        current_signal = {"decision": "NO TRADE", "reason": "No two-sided market"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('crude_orderflow_signal', current_signal)
                        return
                    spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                    if spread_pct > MAX_SPREAD_PCT:
                        current_signal = {"decision": "NO TRADE", "reason": f"Spread too wide ({spread_pct:.1f}%)"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('crude_orderflow_signal', current_signal)
                        return

                    # --- ADDED: simulate a realistic buy fill at the ask, not LTP ---
                    option_ltp = ask
                    if option_ltp <= 48:
                        current_signal = {"decision": "NO TRADE", "reason": f"Option premium too low ({option_ltp})"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('crude_orderflow_signal', current_signal)
                        return
                except Exception as e:
                    logging.warning(f"⚠️ Entry quote/spread check exception for {option_symbol}: {e}")
                    current_signal = {"decision": "NO TRADE",
                                      "reason": f"Exception fetching option quote: {type(e).__name__}"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('crude_orderflow_signal', current_signal)
                    return


                signal_id = str(uuid.uuid4())
                trade_entry_time = now
                entry_option_ltp = option_ltp
                active_trade = {
                    "option_ltp": option_ltp,
                    "highest_premium": option_ltp,
                    "lowest_premium": option_ltp,
                    "bias": bias,
                    "trail_active": False,
                    "breakeven_locked": False,
                    "symbol": option_symbol,
                    "lots": 1,
                    "strike": candidate_strike,
                    "setup_quality": base_score,
                    "signal_quality": signal_quality,
                    "dte": dte,
                    "expiry_date": expiry_date_str,
                    "adx": round(adx_val, 2),
                    "rsi": round(rsi_val, 2),
                    "entry_spread_pct": round(spread_pct, 2),
                    "underlying": "CRUDE",
                    "underlying_ltp": futures_ltp,
                    "fut_sym": fut_sym,
                    "signal_id": signal_id,
                    "market_regime": market_regime,
                    "sl_price": max(option_ltp - 38, 10.0),
                    "feature_scores": convert_numpy({k: v['score'] for k, v in feature_scores.items()}),
                    "trail_distance": max(20, min(55, int(entry_atr * 0.5))),
                    "activation_threshold": max(CRUDE_TRAIL_ACTIVATION, max(20, min(55, int(entry_atr * 0.5)))),
                    "entry_atr": round(entry_atr, 2),
                    "distance_to_level_atr": distance_to_level_atr,
                    "last_quote_time": now,
                    "feature_snapshot": convert_numpy(
                        {k: v['value'] if not isinstance(v, dict) else v for k, v in feature_scores.items()}),
                }
                save_state()

                log_json("TRADE_OPENED", {
                    "signal_id": signal_id,
                    "strategy_version": STRATEGY_VERSION,
                    "entry_time": now.isoformat(),
                    "bias": bias,
                    "strike": candidate_strike,
                    "symbol": option_symbol,
                    "entry_price": option_ltp,
                    "underlying_entry": futures_ltp,
                    "base_score": base_score,
                    "signal_quality": signal_quality,
                    "market_regime": market_regime,
                    "feature_scores": active_trade['feature_scores'],
                    "feature_snapshot": active_trade['feature_snapshot'],
                    "dte": dte,
                    "adx": active_trade.get('adx', 0),
                    "rsi": active_trade.get('rsi', 0),
                    "entry_spread_pct": active_trade.get('entry_spread_pct', 0),
                    "entry_atr": active_trade.get('entry_atr', 0),
                    "distance_to_level_atr": active_trade.get('distance_to_level_atr'),
                    "is_sim": True
                })
                logging.info(f"🔁 [SIM] CRUDE ENTRY: {option_symbol} @ {option_ltp} | Base: {base_score} | Total: {signal_quality} | ID: {signal_id}")

            if active_trade:
                monitor_signal = {
                    "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                    "is_monitoring": True,
                    "symbol": active_trade.get('symbol', ''),
                    "strike": active_trade.get('strike'),
                    "entry_ltp": round(entry_option_ltp, 2),
                    "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                    "option_sl": round(active_trade.get('sl_price', max(entry_option_ltp - 38, 10.0)), 2),
                    "lots": active_trade.get('lots', 1),
                    "setup_quality": active_trade.get('setup_quality', 0),
                    "signal_quality": active_trade.get('signal_quality', 0),
                    "market_regime": active_trade.get('market_regime', ''),
                    "signal_id": active_trade.get('signal_id'),
                    "last_scan": now.strftime("%H:%M:%S"),
                    "primary_reason": f"[SIM] Score {active_trade.get('signal_quality', 0)}",
                    # Full dashboard fields
                    "dte": active_trade.get('dte', 0),
                    "expiry_date": active_trade.get('expiry_date'),
                    "vix_value": 0.0,
                    "spot_price": round(futures_ltp, 2),
                    "scenario": compute_scenario_probs(base_score, bias),
                    "vix_regime": "N/A",
                    "vix_direction": "N/A",
                    "adx": round(adx_val, 2),
                    "momentum_checks": 2 if adx_val > 25 else 1 if adx_val > 20 else 0,
                    "kills": [],
                    "failed_criteria": [],
                    "time_elapsed": round((now - trade_entry_time).total_seconds() / 60, 1) if trade_entry_time else 0,
                    "highest_premium": round(active_trade.get('highest_premium', entry_option_ltp), 2),
                    "trail_stop": round(
                        active_trade.get('highest_premium', entry_option_ltp) - active_trade.get('trail_distance', 0),
                        2)
                    if active_trade.get('trail_active', False) else None,
                    "trail_active": active_trade.get('trail_active', False),
                    "max_loss": round(
                        (entry_option_ltp - active_trade.get('sl_price',
                                                             max(entry_option_ltp - 38, 10.0))) * CRUDE_LOT_SIZE, 0),
                }
                safe_emit('crude_orderflow_signal', monitor_signal)

            # --- FALLBACK: emit NO TRADE if we reached the end without any trade ---
            if active_trade is None:
                current_signal = {
                    "decision": "NO TRADE",
                    "reason": "No entry signal generated",
                    "last_scan": now.strftime("%H:%M:%S"),
                    "dte": dte,
                    "spot_price": round(futures_ltp, 2),
                    "expiry_date": expiry_date_str,
                }
                safe_emit('crude_orderflow_signal', current_signal)
        except Exception as e:
            logging.error(f"Crude order-flow scan error: {e}")

def background_scanner():
    while True:
        run_crude_orderflow_scan()
        time.sleep(60 if active_trade is None else 1)

# ======================== FLASK ROUTES ========================
@app.route('/')
def index():
    return send_from_directory('templates', 'orderflow_crude.html')

@socketio.on('request_signal')
def handle_request():
    with state_lock:
        sig = current_signal.copy()
    safe_emit('crude_orderflow_signal', sig)

@socketio.on('connect')
def handle_connect():
    print("✅ Client connected to Socket.IO")
    with state_lock:
        sig = current_signal.copy()
    emit('crude_orderflow_signal', sig)

@socketio.on('disconnect')
def handle_disconnect():
    print("⚠️ Client disconnected from Socket.IO")

@socketio.on('force_exit')
def handle_force_exit():
    global current_signal
    with state_lock:
        if active_trade is not None:
            exit_pnl = force_close_trade("MANUAL EXIT", "MANUAL FORCE CLOSE", 0, is_sim=True)
            now = now_ist()
            current_signal = {
                "decision": "EXIT — MANUAL",
                "reason": f"Manual exit | PnL: ₹{exit_pnl:.0f}",
                "last_scan": now.strftime("%H:%M:%S")
            }
            safe_emit('crude_orderflow_signal', current_signal)
            logging.info(f"Manual exit triggered - PnL: ₹{exit_pnl:.0f}")
        else:
            logging.info("Manual exit requested but no active trade")

if __name__ == "__main__":
    load_state()
    threading.Thread(target=background_scanner, daemon=True).start()
    print(f"🚀 Crude Order-Flow Engine (SHADOW) v{STRATEGY_VERSION} running on http://localhost:{PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False)