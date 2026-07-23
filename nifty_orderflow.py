# === NIFTY ORDER-FLOW BUYER ENGINE v2.8 (SHADOW MODE) ===
# v2.8:
#   - Full dashboard fields added (dte, vix, spot, scenario, adx, momentum, etc.)
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
from collections import deque

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

PORT = 8063
MAX_LOTS = 1
NIFTY_LOT_SIZE = 65
STATE_FILE = "nifty_orderflow_state.json"
NIFTY_TRAIL_ACTIVATION = 8
NIFTY_BREAKEVEN_PCT = 0.15   # lock breakeven once profit hits 15% of entry premium
NIFTY_TRAIL_FLOOR = 6        # tightest the trail can ratchet down to
NIFTY_SL_PCT = 0.30          # SL = 30% of entry premium (replaces fixed ₹30 SL)
NIFTY_DEAD_TRADE_MINUTES = 18   # exit if trail never activates within this many minutes (cuts slow-bleed losers)
NIFTY_DEAD_TRADE_MINUTES_ATM = 10   # tighter leash for DTE<=2 ATM trades — faster theta bleed, no OTM buffer
MAX_SPREAD_PCT = 5.0
HTF_MISMATCH_PENALTY = 15   # points deducted when 1H VWAP disagrees with entry bias
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

STRATEGY_VERSION = "v2.8"

VOLATILITY_THRESHOLD_HIGH = 1.5
VOLATILITY_THRESHOLD_MODERATE = 0.8

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "nifty_orderflow.log"),
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
ENTRY_COOLDOWN_SECONDS = 300
daily_reset_date = now_ist().date()


prev_oi_history = {}  # oi_key -> deque[(timestamp, total_oi)]
OI_THRESHOLD_PCT = 0.005       # 0.5% change required
OI_LOOKBACK_MINUTES = 5

_candle_cache = {}
cached_candles_5m = pd.DataFrame()
cached_candles_15m = pd.DataFrame()
cached_daily_candles = pd.DataFrame()
nfo_instruments_cache = None
nfo_instruments_cache_time = None
last_chain_time = None
cached_chain = pd.DataFrame()
last_chain_expiry = None
cached_ht_candles = pd.DataFrame()
last_ht_candles_time = None


NIFTY_SPOT_TOKEN = 256265
NIFTY_SPOT_SYMBOL = "NSE:NIFTY 50"
VIX_SYMBOL = "NSE:INDIA VIX"
VIX_TOKEN = 264969

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
    json_log_path = os.path.join(LOG_DIR, "nifty_orderflow.jsonl")
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
    score = min(20, max(0, (rvol - 0.8) * 25))
    return {"value": rvol, "score": round(score, 2), "reason": f"RVOL {rvol:.2f}"}


def compute_oi_acceleration(oi_dq, lookback_minutes=None):
    """
    Second derivative of ATM (CE+PE) OI over time, using the same
    time-anchored lookback as oi_chg — reads timestamps, not array
    position, so it's stable whether scans run every 60s or every 1s.
    """
    lookback = lookback_minutes or OI_LOOKBACK_MINUTES
    if oi_dq is None or len(oi_dq) < 3:
        return {"value": 0, "score": 0, "reason": "insufficient OI history"}

    now_ts, oi_now = oi_dq[-1]
    t1 = now_ts - datetime.timedelta(minutes=lookback)
    t2 = now_ts - datetime.timedelta(minutes=lookback * 2)

    oi_1 = next((oi for ts, oi in reversed(oi_dq) if ts <= t1), None)
    oi_2 = next((oi for ts, oi in reversed(oi_dq) if ts <= t2), None)

    if oi_1 is None or oi_2 is None:
        return {"value": 0, "score": 0, "reason": f"insufficient OI history (need {lookback*2}min)"}

    diff1 = oi_now - oi_1
    diff2 = oi_1 - oi_2
    accel = diff1 - diff2

    # UNCALIBRATED divisor — ATM CE+PE OI moves on a different scale
    # than Crude's futures OI. Log raw `accel` for a week before trusting
    # the score magnitude; adjust the divisor from that distribution.
    score = min(18, max(-10, accel / 20000 * 2))
    return {"value": accel, "score": round(score, 2), "reason": f"OI accel {accel:.0f}"}

def compute_value_area(candles, period=20):
    if len(candles) < period:
        return {"poc": None, "vah": None, "val": None, "score": 0, "reason": "insufficient data"}
    df = candles.tail(period).copy()
    bins = pd.cut(df['close'], bins=20, precision=1)
    vol_by_bin = df.groupby(bins, observed=False)['volume'].sum()
    poc_bin = vol_by_bin.idxmax()
    poc = poc_bin.mid if hasattr(poc_bin, 'mid') else None
    sorted_bins = vol_by_bin.sort_values(ascending=False)
    cum_vol = 0
    target = vol_by_bin.sum() * 0.7
    vah, val = None, None
    for b, vol in sorted_bins.items():
        cum_vol += vol
        if cum_vol >= target:
            vah = b.right if hasattr(b, 'right') else b
            break
    cum_vol = 0
    for b, vol in sorted_bins.items():
        cum_vol += vol
        if cum_vol >= target:
            val = b.left if hasattr(b, 'left') else b
            break
    price = candles['close'].iloc[-1]
    score = 0
    reason = ""
    if poc is not None:
        if price > poc * 0.995 and price < poc * 1.005:
            score = 15
            reason = "Price near POC"
        else:
            score = 5
            reason = "Price away from POC"
    return {"poc": poc, "vah": vah, "val": val, "score": score, "reason": reason}

def compute_strike_rotation(chain_df, spot):
    if chain_df.empty:
        return {"value": 0, "score": 0, "reason": "no chain"}
    atm = round(spot / 50) * 50
    strikes = chain_df.groupby('strike')['oi'].sum()
    if strikes.empty:
        return {"value": 0, "score": 0, "reason": "no OI"}
    max_oi_strike = strikes.idxmax()
    rotation = max_oi_strike - atm
    score = 10 if abs(rotation) >= 50 else 0
    reason = f"Strike rotation {rotation}" if abs(rotation) >= 50 else "No significant rotation"
    return {"value": rotation, "score": score, "reason": reason}

def compute_option_wall(chain_df, option_type):
    if chain_df.empty:
        return {"strike": None, "oi": 0, "score": 0, "reason": "no chain"}
    wall_df = chain_df[chain_df['instrument_type'] == option_type]
    if wall_df.empty:
        return {"strike": None, "oi": 0, "score": 0, "reason": "no options"}
    max_oi_row = wall_df.loc[wall_df['oi'].idxmax()]
    strike = max_oi_row['strike']
    oi = max_oi_row['oi']
    score = 5
    reason = f"Option wall at {strike}"
    return {"strike": strike, "oi": oi, "score": score, "reason": reason}

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
        return {"value": 1, "score": 10, "reason": "Breakout above PDH accepted"}
    elif low < pdl and close < pdl * 1.002:
        return {"value": 1, "score": 10, "reason": "Breakout below PDL accepted"}
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
    va = feature_scores.get("value_area", {}).get("score", 0)
    breakout = feature_scores.get("breakout_acceptance", {}).get("value", 0)
    strike_rot = feature_scores.get("strike_rotation", {}).get("value", 0)

    if rvol > 1.5 and va > 10:
        bonus += 5
    if breakout == 1 and abs(strike_rot) > 50:
        bonus += 5
    if rvol > 1.3 and abs(strike_rot) > 30:
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
                           spot_ltp, market_regime, dte, reason):
    """Append score details to logs/nifty_score_distribution.csv on every scan."""
    score_log_path = os.path.join(LOG_DIR, "nifty_score_distribution.csv")
    try:
        with _score_lock:
            write_header = not os.path.exists(score_log_path)
            with open(score_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "total_score", "base_score", "bonus", "interaction_bonus",
                        "bias", "spot_ltp", "market_regime", "dte", "reason"
                    ])
                writer.writerow([
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    round(total_score, 2),
                    round(base_score, 2),
                    round(bonus, 2),
                    round(interaction_bonus, 2),
                    bias,
                    round(spot_ltp, 2),
                    market_regime,
                    dte,
                    reason
                ])
    except Exception as e:
        logging.warning(f"Score distribution log failed: {e}")

# ======================== COMPOSITE SCORE ========================
def composite_score(candles, price_chg, oi_chg, key_levels, volume_candles=None):
    reasons = []
    score = 0
    bias = "NEUTRAL"

    # --- ADDED: swap in futures volume, keep spot price/levels untouched ---
    if volume_candles is not None and len(volume_candles) == len(candles):
        candles = candles.copy()
        candles['volume'] = volume_candles['volume'].values
    # --- END ADDED ---

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
        vol_by_bin = candles.groupby(bins, observed=False)['volume'].sum()  # ✅ use candles
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
            atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

            dm_plus = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
            dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)

            # ⬇️ FIX: atr is a scalar – use a conditional instead of .replace()
            if atr == 0:
                atr = 1.0

            di_plus = 100 * dm_plus.ewm(alpha=1 / 14, adjust=False).mean() / atr
            di_minus = 100 * dm_minus.ewm(alpha=1 / 14, adjust=False).mean() / atr

            # ⬇️ FIX: denom is a Series – .replace() is safe here
            denom = (di_plus + di_minus).replace(0, 1)
            dx = 100 * (di_plus - di_minus).abs() / denom
            adx = dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

            # ⬇️ FIX: loss is a scalar – handle zero separately
            if loss == 0:
                rs = 999999  # effectively infinite → RSI = 100
            else:
                rs = gain / loss

            rsi = 100 - (100 / (1 + rs))

            if 20 <= adx <= 27:
                score += 10
                reasons.append("Moderate trend momentum")
            elif adx > 30:
                score -= 10
                reasons.append("ADX overextended – late entry risk")
            elif adx < 20:
                score -= 5
                reasons.append("Weak trend – cautious")


    score = max(0, min(100, score))
    if bias == "NEUTRAL":
        bias = "CALL" if score > 50 else "PUT" if score > 40 else "NEUTRAL"
    return {"score": score, "bias": bias, "oi_class": oi_class, "reasons": reasons}

# ======================== NIFTY HELPERS ========================
def get_nfo_instruments(force=False):
    global nfo_instruments_cache, nfo_instruments_cache_time
    now = now_ist()
    if not force and nfo_instruments_cache_time and (now - nfo_instruments_cache_time).seconds < 3600:
        return nfo_instruments_cache
    try:
        nfo_instruments_cache = kite_call_with_timeout(kite.instruments, "NFO")
        if nfo_instruments_cache is None:
            nfo_instruments_cache = []
        nfo_instruments_cache_time = now
        return nfo_instruments_cache
    except Exception as e:
        logging.error(f"NFO instruments fetch error: {e}")
        return nfo_instruments_cache or []

def resolve_weekly_expiry():
    try:
        nfo = get_nfo_instruments(force=False)
        nifty_options = [i for i in nfo if i.get('name') == 'NIFTY' and i.get('instrument_type') in ['CE', 'PE']]
        if not nifty_options:
            return None
        today = now_ist().date()
        expiries = set()
        for opt in nifty_options:
            exp = opt.get('expiry')
            if exp:
                if isinstance(exp, str):
                    exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
                else:
                    exp_date = exp.date() if hasattr(exp, 'date') else exp
                if exp_date >= today:
                    expiries.add(exp_date)
        if not expiries:
            return None
        nearest = min(expiries)
        logging.info(f"✅ [SIM] Resolved expiry: {nearest}")
        return nearest
    except Exception as e:
        logging.error(f"Weekly expiry resolution error: {e}")
        return None

def get_candles(token, interval="5minute", days=2, force=False):
    global _candle_cache
    now = now_ist()
    cache_key = (token, interval)
    cache_timeout = {"5minute": 60, "15minute": 120, "day": 300, "60minute": 180}.get(interval, 60)
    if not force and cache_key in _candle_cache:
        cached_time, cached_df = _candle_cache[cache_key]
        if (now - cached_time).seconds < cache_timeout:
            return cached_df
    try:
        to_date = now
        from_date = to_date - datetime.timedelta(days=days)
        data = kite_call_with_timeout(kite.historical_data, token, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"), interval)
        if data is None:
            data = []
        df = pd.DataFrame(data)
        _candle_cache[cache_key] = (now, df)
        return df
    except Exception as e:
        logging.warning(f"Historical data error: {e}")
        return _candle_cache.get(cache_key, (None, pd.DataFrame()))[1] if cache_key in _candle_cache else pd.DataFrame()

def fetch_option_chain(expiry_date, spot_ltp, vix_ltp):
    global last_chain_time, cached_chain, last_chain_expiry
    now = now_ist()
    if (last_chain_time and (now - last_chain_time).seconds < 120 and not cached_chain.empty and last_chain_expiry == expiry_date):
        return cached_chain
    try:
        nfo = get_nfo_instruments()
        expiry_str = expiry_date.strftime('%Y-%m-%d')
        range_width = 500 if vix_ltp > 20 else 300
        atm = round(spot_ltp / 100) * 100
        min_strike = atm - range_width
        max_strike = atm + range_width
        options = []
        for inst in nfo:
            if inst.get('name') == 'NIFTY' and inst.get('instrument_type') in ['CE', 'PE']:
                exp = inst.get('expiry')
                if exp is None:
                    continue
                if isinstance(exp, str):
                    inst_expiry_str = exp
                elif hasattr(exp, 'strftime'):
                    inst_expiry_str = exp.strftime('%Y-%m-%d')
                else:
                    inst_expiry_str = str(exp)
                if inst_expiry_str == expiry_str:
                    strike = inst.get('strike', 0)
                    if min_strike <= strike <= max_strike:
                        options.append(inst)
        if len(options) < 10:
            return pd.DataFrame()
        symbols = [f"NFO:{opt['tradingsymbol']}" for opt in options]
        quotes = {}
        failed_batches = 0
        for i in range(0, len(symbols), 200):
            batch = symbols[i:i+200]
            try:
                batch_quotes = kite_call_with_timeout(kite.quote, batch)
                if batch_quotes is not None:
                    quotes.update(batch_quotes)
                else:
                    failed_batches += 1
            except Exception as e:
                failed_batches += 1
                logging.warning(f"⚠️ Option chain batch fetch failed (batch {i}-{i + 200}): {e}")

        if failed_batches > 0:
            logging.warning(
                f"⚠️ Option chain fetch incomplete — {failed_batches} of {(len(symbols) + 199) // 200} batches failed; Max Pain/PCR/walls computed on partial chain")

        chain_data = []
        for opt in options:
            sym = f"NFO:{opt['tradingsymbol']}"
            q = quotes.get(sym, {})
            if q:
                depth_buy = q.get('depth', {}).get('buy', [{}])
                depth_sell = q.get('depth', {}).get('sell', [{}])
                bid = depth_buy[0].get('price', 0) if depth_buy else 0
                ask = depth_sell[0].get('price', 0) if depth_sell else 0
                chain_data.append({
                    'tradingsymbol': opt['tradingsymbol'],
                    'strike': opt['strike'],
                    'instrument_type': opt['instrument_type'],
                    'expiry': opt.get('expiry').strftime('%Y-%m-%d') if hasattr(opt.get('expiry'), 'strftime') else opt.get('expiry'),
                    'ltp': q.get('last_price', 0),
                    'oi': q.get('oi', 0),
                    'volume': q.get('volume', 0),
                    'bid': bid,
                    'ask': ask,
                })
        df = pd.DataFrame(chain_data)
        if not df.empty:
            cached_chain = df
            last_chain_time = now
            last_chain_expiry = expiry_date
        return df
    except Exception as e:
        logging.error(f"Option chain fetch error: {e}")
        if not cached_chain.empty and last_chain_expiry == expiry_date:
            return cached_chain
        return pd.DataFrame()

def resolve_nifty_futures():
    try:
        nfo = get_nfo_instruments()
        now = now_ist()
        year_short = str(now.year)[-2:]
        month_abbr = calendar.month_abbr[now.month].upper()
        tradingsymbol = f"NIFTY{year_short}{month_abbr}FUT"
        matches = [i for i in nfo if i.get('tradingsymbol') == tradingsymbol and i.get('name') == 'NIFTY']
        if matches:
            return matches[0]
        next_month = now.month + 1 if now.month < 12 else 1
        next_year = now.year if now.month < 12 else now.year + 1
        year_short = str(next_year)[-2:]
        month_abbr = calendar.month_abbr[next_month].upper()
        tradingsymbol = f"NIFTY{year_short}{month_abbr}FUT"
        matches = [i for i in nfo if i.get('tradingsymbol') == tradingsymbol and i.get('name') == 'NIFTY']
        if matches:
            return matches[0]
        return None
    except Exception as e:
        logging.error(f"Futures resolution error: {e}")
        return None

def get_vix_regime(vix_ltp):
    if vix_ltp < 13: return "Strong Sell Premium"
    elif vix_ltp < 17: return "Neutral"
    elif vix_ltp <= 22: return "Elevated/Favour Buying"
    else: return "Fear/Long Strangle bias"

def get_vix_direction(vix_history):
    if vix_history.empty or len(vix_history) < 3: return "Flat"
    last_3 = vix_history['close'].tail(3)
    if last_3.iloc[-1] > last_3.iloc[0] * 1.02: return "Rising"
    elif last_3.iloc[-1] < last_3.iloc[0] * 0.98: return "Falling"
    return "Flat"

def safe_emit(event, data):
    try:
        socketio.emit(event, data)
        logging.info(f"EMIT {event}: {data.get('decision')} | {data.get('reason')}")
    except Exception as e:
        logging.exception(f"safe_emit FAILED for {event}: {e}")
        print(f"❌ safe_emit FAILED: {e}")

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


def calculate_vwap(df):
    if df.empty or 'volume' not in df.columns or df['volume'].sum() == 0:
        return 0
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
            # --- FIX: if volume is zero, vwap becomes NaN – treat as missing ---
            if pd.isna(vwap.iloc[-1]):
                vwap_val = None
            else:
                vwap_val = round(vwap.iloc[-1], 2)
        else:
            vwap_val = None
    else:
        vwap_val = None
    return {"PDH": pdh, "PDL": pdl, "VWAP": vwap_val}

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
            q = kite_call_with_timeout(kite.quote, [f"NFO:{active_trade['symbol']}"])
            if q is None:
                q = {}
            depth = q.get(f"NFO:{active_trade['symbol']}", {}).get('depth', {})
            fresh_bid = depth.get('buy', [{}])[0].get('price', 0)
            if fresh_bid > 0:
                exit_ltp = fresh_bid  # simulate a realistic sell fill at the bid
            else:
                fresh = q.get(f"NFO:{active_trade['symbol']}", {}).get('last_price')
                if fresh and fresh > 0:
                    exit_ltp = fresh
                    logging.warning(f"⚠️ NIFTY no bid price for {active_trade.get('symbol')}, falling back to LTP")
        except Exception as e:
            logging.warning(
                f"⚠️ NIFTY fresh exit-price fetch failed for {active_trade.get('symbol')} — using last known price {exit_ltp} for PnL calc: {e}")

    lots = active_trade.get('lots', 1)
    entry = entry_option_ltp
    exit_pnl = (exit_ltp - entry) * lots * NIFTY_LOT_SIZE
    daily_pnl += exit_pnl

    highest = active_trade.get('highest_premium', entry)
    lowest = active_trade.get('lowest_premium', entry)
    mfe_pts = (highest - entry) * lots * NIFTY_LOT_SIZE
    mae_pts = max(0, (entry - lowest) * lots * NIFTY_LOT_SIZE)

    sl_price = active_trade.get('sl_price', entry * (1 - NIFTY_SL_PCT))
    risk_per_lot = abs(entry - sl_price) * NIFTY_LOT_SIZE
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
        "rsi": active_trade.get('rsi', 0),
        "entry_atr": active_trade.get('entry_atr', 0),
        "vix_value": active_trade.get('vix_value', 0),
        "feature_scores": active_trade.get('feature_scores'),
        "dte": active_trade.get('dte'),
        "dead_trade_minutes": active_trade.get('dead_trade_minutes'),
        "adx": active_trade.get('adx'),
        "is_sim": is_sim
    })

    try:
        regime = active_trade.get('market_regime', 'UNKNOWN')
        regime_file = os.path.join(LOG_DIR, "nifty_regime_performance.csv")
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
def run_nifty_orderflow_scan():
    global current_signal, trade_entry_time, entry_option_ltp, active_trade, daily_pnl, daily_reset_date
    print(f"🔍 Nifty scan running at {now_ist().strftime('%H:%M:%S')}")
    with state_lock:
        try:
            now = now_ist()
            if daily_reset_date is None:
                daily_reset_date = now_ist().date()

            market_open = now.replace(hour=9, minute=45, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=15, second=0, microsecond=0)
            if not (market_open <= now <= market_close):
                if active_trade:
                    force_close_trade("Market Closed", "MARKET CLOSE", None, is_sim=True)
                # Emit a NO TRADE signal so the dashboard updates
                current_signal = {"decision": "NO TRADE", "reason": "Market Closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('nifty_orderflow_signal', current_signal)
                print(f"🔴 REJECTED: Market Closed") # delete it for debugging only
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
                        opt_quote = kite_call_with_timeout(kite.quote, [f"NFO:{opt_symbol}"])
                        if opt_quote is not None:
                            depth = opt_quote.get(f"NFO:{opt_symbol}", {}).get('depth', {})
                            ltp = opt_quote.get(f"NFO:{opt_symbol}", {}).get('last_price', 0)
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
                                    safe_emit('nifty_orderflow_signal', current_signal)
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
                                safe_emit('nifty_orderflow_signal', current_signal)
                                return
                    except Exception as e:
                        logging.warning(
                            f"⚠️ NIFTY option premium refresh exception for {opt_symbol}: {e} – treating as stale"
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
                            safe_emit('nifty_orderflow_signal', current_signal)
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

                # --- BREAKEVEN LOCK: once MFE clears NIFTY_BREAKEVEN_PCT of entry premium, SL -> entry ---
                if not active_trade.get('breakeven_locked', False):
                    breakeven_trigger = entry_option_ltp * NIFTY_BREAKEVEN_PCT
                    if highest_premium - entry_option_ltp >= breakeven_trigger:
                        active_trade['sl_price'] = max(
                            active_trade.get('sl_price', entry_option_ltp * (1 - NIFTY_SL_PCT)),
                                                       entry_option_ltp)
                        active_trade['breakeven_locked'] = True
                        logging.info(
                            f"🔒 [SIM] NIFTY Breakeven lock engaged | entry {entry_option_ltp} | peak {highest_premium}")

                # Fetch underlying LTP for logging (non‑blocking)
                underlying_ltp = active_trade.get('underlying_ltp', 0)
                try:
                    q = kite_call_with_timeout(kite.quote, [NIFTY_SPOT_SYMBOL])
                    if q:
                        underlying_ltp = q.get(NIFTY_SPOT_SYMBOL, {}).get('last_price', underlying_ltp)
                except Exception as e:
                    logging.warning(f"⚠️ NIFTY underlying LTP refresh failed (using stale price {underlying_ltp}): {e}")

                # --- Fixed SL: only apply if trail is NOT active ---
                if not active_trade.get('trail_active', False):
                    sl_price = active_trade.get('sl_price', max(entry_option_ltp * (1 - NIFTY_SL_PCT), 10.0))
                    active_trade['sl_price'] = sl_price
                    if current_premium <= sl_price:
                        exit_pnl = force_close_trade(f"SL HIT (₹{abs(entry_option_ltp - sl_price):.0f})", "STOP LOSS",
                                                     underlying_ltp, is_sim=True)
                        current_signal = {"decision": "EXIT — STOP LOSS", "reason": f"SL hit | PnL: ₹{exit_pnl:.0f}"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        return

                # --- PROFIT-RATCHETING TRAIL: tighten trail distance as MFE grows, floor at NIFTY_TRAIL_FLOOR ---
                base_trail = active_trade.get('trail_distance', 8)
                mfe = highest_premium - entry_option_ltp
                if mfe >= base_trail * 2.5:
                    trail_distance = max(NIFTY_TRAIL_FLOOR, base_trail * 0.5)
                elif mfe >= base_trail * 1.5:
                    trail_distance = max(NIFTY_TRAIL_FLOOR, base_trail * 0.7)
                else:
                    trail_distance = base_trail

                active_trade['trail_distance'] = trail_distance
                activation_threshold = active_trade.get('activation_threshold', NIFTY_TRAIL_ACTIVATION)

                if not active_trade.get('trail_active', False):
                    if current_premium >= entry_option_ltp + activation_threshold:
                        active_trade['trail_active'] = True

                # --- TRAILING STOP EXIT (only if trail is active) ---
                if active_trade.get('trail_active', False):
                    if highest_premium - current_premium >= trail_distance:
                        exit_pnl = force_close_trade(
                            f"TRAILING STOP (peak {round(highest_premium, 2)}, exit {round(current_premium, 2)})",
                            "TRAILING STOP", underlying_ltp, is_sim=True)
                        current_signal = {"decision": "EXIT — TRAILING STOP",
                                          "reason": f"Peak {round(highest_premium, 2)} → {round(current_premium, 2)} | PnL: ₹{exit_pnl:.0f}"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        return


                # --- DEAD-TRADE CUT: if trail never activated within the trade's window, exit ---
                # (checked AFTER trail-activation so a trade that just crossed the threshold
                # this tick isn't force-closed on a stale trail_active=False read)
                if not active_trade.get('trail_active', False):
                    minutes_in_trade = (now - trade_entry_time).total_seconds() / 60
                    dead_trade_limit = active_trade.get('dead_trade_minutes', NIFTY_DEAD_TRADE_MINUTES)
                    if minutes_in_trade >= dead_trade_limit:
                        exit_pnl = force_close_trade(
                            f"DEAD TRADE CUT ({minutes_in_trade:.1f}m / {dead_trade_limit}m limit)",
                            "DEAD TRADE", underlying_ltp, is_sim=True)
                        current_signal = {"decision": "EXIT — DEAD TRADE",
                                          "reason": f"No trail activation in {dead_trade_limit}m | PnL: ₹{exit_pnl:.0f}"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        return

            # === COOLDOWN: wait after an exit before re-entering ===
            if active_trade is None and last_exit_time:
                elapsed = (now - last_exit_time).total_seconds()
                if elapsed < ENTRY_COOLDOWN_SECONDS:
                    remaining = int(ENTRY_COOLDOWN_SECONDS - elapsed)
                    current_signal = {"decision": "NO TRADE", "reason": f"Cooldown ({remaining}s remaining)"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('nifty_orderflow_signal', current_signal)
                    print(f"🔴 REJECTED: Cooldown ({remaining}s remaining)")
                    return

            # === STEP 2: FETCH SPOT/VIX for entry signals ===
            try:
                spot_quote = kite_call_with_timeout(kite.quote, [NIFTY_SPOT_SYMBOL, VIX_SYMBOL])
                if spot_quote is None:
                    spot_quote = {}
                spot_ltp = spot_quote.get(NIFTY_SPOT_SYMBOL, {}).get('last_price', 0)
                vix_ltp = spot_quote.get(VIX_SYMBOL, {}).get('last_price', 0)
                if spot_ltp <= 0:
                    current_signal = {"decision": "NO TRADE", "reason": "Failed to fetch spot/VIX"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('nifty_orderflow_signal', current_signal)
                    print(f"🔴 REJECTED: Spot/VIX fetch failed") # delete for debugging purpose
                    return
            except Exception as e:
                logging.warning(f"⚠️ NIFTY spot/VIX fetch exception: {e}")
                current_signal = {"decision": "NO TRADE", "reason": f"Exception fetching spot/VIX: {type(e).__name__}"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('nifty_orderflow_signal', current_signal)
                print(f"🔴 REJECTED: Spot/VIX fetch failed")  # delete for debugging purpose
                return

            # === STEP 3: ENTRY SIGNAL COMPUTATION ===
            if daily_pnl <= max_daily_loss:
                current_signal = {
                    "decision": "NO TRADE",
                    "reason": f"Daily loss cap reached",
                    "last_scan": now.strftime("%H:%M:%S"),
                    "dte": dte if 'dte' in locals() else 0,
                    "vix_value": round(vix_ltp, 2) if 'vix_ltp' in locals() else 0,
                    "spot_price": round(spot_ltp, 2) if 'spot_ltp' in locals() else 0,
                    "scenario": compute_scenario_probs(0, "NEUTRAL"),
                    "vix_regime": vix_regime if 'vix_regime' in locals() else "N/A",
                    "vix_direction": vix_direction if 'vix_direction' in locals() else "N/A",
                    "adx": 0,
                    "momentum_checks": 0,
                    "kills": [],
                    "failed_criteria": [],
                    "time_elapsed": 0,
                    "highest_premium": 0,
                    "trail_stop": 0,
                    "setup_quality": 0,
                    "signal_quality": 0,
                    "market_regime": "NEUTRAL",
                }
                safe_emit('nifty_orderflow_signal', current_signal)
                if active_trade:
                    monitor_signal = {
                        "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                        "is_monitoring": True,
                        "strike": active_trade.get('strike'),
                        "symbol": active_trade.get('symbol', ''),
                        "entry_ltp": round(entry_option_ltp, 2),
                        "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                        "bid_price": round(active_trade.get('trail_premium', active_trade.get('option_ltp', 0)), 2),
                        "option_sl": round(
                            active_trade.get('sl_price', max(entry_option_ltp * (1 - NIFTY_SL_PCT), 10.0)), 2),
                        "lots": active_trade.get('lots', 1),
                        "setup_quality": active_trade.get('setup_quality', 0),
                        "signal_quality": active_trade.get('signal_quality', 0),
                        "market_regime": active_trade.get('market_regime', ''),
                        "signal_id": active_trade.get('signal_id'),
                        "last_scan": now.strftime("%H:%M:%S"),
                        "primary_reason": "[SIM] Monitoring only (loss cap reached)",
                    }
                    safe_emit('nifty_orderflow_signal', monitor_signal)
                print(f"🔴 REJECTED: Daily loss cap reached (PnL: {daily_pnl})") # for debugging purpose
                return

            candles_5m = get_candles(NIFTY_SPOT_TOKEN, "5minute", 5)
            candles_15m = get_candles(NIFTY_SPOT_TOKEN, "15minute", 5)
            candles_daily = get_candles(NIFTY_SPOT_TOKEN, "day", 60)

            # --- ADDED: index has no real traded volume; use futures for volume-based features ---
            fut_instrument = resolve_nifty_futures()
            if fut_instrument:
                candles_5m_vol = get_candles(fut_instrument['instrument_token'], "5minute", 5)
                candles_15m_vol = get_candles(fut_instrument['instrument_token'], "15minute", 5)
            else:
                candles_5m_vol = candles_5m
                candles_15m_vol = candles_15m
                logging.warning(
                    "⚠️ Nifty futures resolution failed — RVOL/Value Area/POC falling back to spot candles (volume likely zero)")

            # --- ADDED: index-price candles with futures volume substituted in, for VWAP/key-level use ---
            if len(candles_5m_vol) == len(candles_5m):
                candles_5m_for_vwap = candles_5m.copy()
                candles_5m_for_vwap['volume'] = candles_5m_vol['volume'].values
            else:
                candles_5m_for_vwap = candles_5m
                logging.warning(
                    "⚠️ Nifty spot/futures candle length mismatch — VWAP/key-levels falling back to spot candles (volume likely zero)")

            expiry_date = resolve_weekly_expiry()
            if not expiry_date:
                current_signal = {
                    "decision": "NO TRADE",
                    "reason": "Expiry resolution failed",
                    "last_scan": now.strftime("%H:%M:%S"),
                    "dte": 0,
                    "vix_value": round(vix_ltp, 2),
                    "spot_price": round(spot_ltp, 2),
                    "scenario": {"upside": 0, "downside": 0, "flat": 0},
                    "vix_regime": "N/A",
                    "vix_direction": "N/A",
                    "adx": 0,
                    "momentum_checks": 0,
                    "kills": [],
                    "failed_criteria": [],
                    "time_elapsed": 0,
                    "highest_premium": 0,
                    "trail_stop": 0,
                    "setup_quality": 0,
                    "signal_quality": 0,
                    "market_regime": "NEUTRAL",
                }
                safe_emit('nifty_orderflow_signal', current_signal)
                print(f"🔴 REJECTED: Expiry resolution failed") # for debugging purpose
                return
            dte = (expiry_date - now_ist().date()).days

            if active_trade is None:
                cutoff = None
                if dte == 0:
                    cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
                elif dte == 1:
                    cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
                if cutoff and now >= cutoff:
                    current_signal = {"decision": "NO TRADE",
                                      "reason": f"No entries on DTE {dte} after {cutoff.strftime('%H:%M')}"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    current_signal["expiry_date"] = expiry_date.isoformat()
                    safe_emit('nifty_orderflow_signal', current_signal)
                    print(f"🔴 REJECTED: DTE {dte} cutoff passed")
                    return

            vix_history = get_candles(VIX_TOKEN, "day", 5)
            vix_regime = get_vix_regime(vix_ltp)
            vix_direction = get_vix_direction(vix_history)

            entry_atr = 0
            if not candles_5m.empty and len(candles_5m) >= 14:
                tr = pd.concat([
                    candles_5m['high'] - candles_5m['low'],
                    (candles_5m['high'] - candles_5m['close'].shift()).abs(),
                    (candles_5m['low'] - candles_5m['close'].shift()).abs()
                ], axis=1).max(axis=1)
                entry_atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

            vwap = 0
            if len(candles_5m_for_vwap) >= 3:
                todays_candles = candles_5m_for_vwap[candles_5m_for_vwap[
                                                         'date'].dt.date == now_ist().date()] if 'date' in candles_5m_for_vwap.columns else candles_5m_for_vwap
                vwap = calculate_vwap(todays_candles) if not todays_candles.empty else spot_ltp

            key_levels = get_key_levels(candles_daily, candles_5m_for_vwap)

            chain_df = fetch_option_chain(expiry_date, spot_ltp, vix_ltp)
            if chain_df.empty:
                current_signal = {
                    "decision": "NO TRADE",
                    "reason": "Option chain empty",
                    "last_scan": now.strftime("%H:%M:%S"),
                    "expiry_date": expiry_date.isoformat(),
                    "dte": dte,
                    "vix_value": round(vix_ltp, 2),
                    "spot_price": round(spot_ltp, 2),
                    "scenario": {"upside": 0, "downside": 0, "flat": 0},
                    "vix_regime": vix_regime,
                    "vix_direction": vix_direction,
                    "adx": 0,
                    "momentum_checks": 0,
                    "kills": [],
                    "failed_criteria": [],
                    "time_elapsed": 0,
                    "highest_premium": 0,
                    "trail_stop": 0,
                    "setup_quality": 0,
                    "signal_quality": 0,
                    "market_regime": "NEUTRAL",
                }
                safe_emit('nifty_orderflow_signal', current_signal)
                print(f"🔴 REJECTED: Option chain empty")
                return

            # AFTER — reordered so OI history is updated before it's read
            price_chg = candles_15m['close'].iloc[-1] - candles_15m['close'].iloc[-2] if len(candles_15m) >= 2 else 0

            atm_strike = round(spot_ltp / 100) * 100
            atm_ce = chain_df[(chain_df['strike'] == atm_strike) & (chain_df['instrument_type'] == 'CE')]['oi'].sum()
            atm_pe = chain_df[(chain_df['strike'] == atm_strike) & (chain_df['instrument_type'] == 'PE')]['oi'].sum()
            total_oi = atm_ce + atm_pe

            oi_key = f"{expiry_date}_{atm_strike}"
            if oi_key not in prev_oi_history:
                prev_oi_history[oi_key] = deque(maxlen=1200)
            dq = prev_oi_history[oi_key]
            dq.append((now, total_oi))

            oi_chg = 0
            target_time = now - datetime.timedelta(minutes=OI_LOOKBACK_MINUTES)
            past_oi = next((oi for ts, oi in dq if ts <= target_time), None)
            if past_oi and past_oi > 0:
                raw_chg = total_oi - past_oi
                if abs(raw_chg) / past_oi > OI_THRESHOLD_PCT:
                    oi_chg = raw_chg

            rvol = compute_rvol(candles_5m_vol)
            oi_acc = compute_oi_acceleration(dq)
            va = compute_value_area(candles_5m_vol)
            strike_rot = compute_strike_rotation(chain_df, spot_ltp)
            call_wall = compute_option_wall(chain_df, "CE")
            put_wall = compute_option_wall(chain_df, "PE")
            breakout = compute_breakout_acceptance(candles_5m, key_levels)

            feature_scores = {
                "rvol": rvol,
                "oi_acceleration": oi_acc,
                "value_area": va,
                "strike_rotation": strike_rot,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "breakout_acceptance": breakout
            }

            comp = composite_score(candles_15m, price_chg, oi_chg, key_levels, volume_candles=candles_15m_vol)
            base_score = comp["score"]
            bias = comp["bias"]

            bonus = 0
            interaction_bonus = compute_interaction_bonus(feature_scores)
            total_score = base_score + bonus + interaction_bonus

            market_regime = compute_market_regime(candles_5m, entry_atr, vwap)
            signal_quality = min(100, total_score)

            # --- Log score distribution (only when score > 40 to keep file small) ---
            if total_score > 40:
                log_score_distribution(
                    now, total_score, base_score, bonus, interaction_bonus, bias,
                    spot_ltp, market_regime, dte,
                    "Accepted" if total_score >= 52 and bias != "NEUTRAL" else "Rejected"
                )

            # Compute ADX for logging
            adx_val = 0
            rsi_val = 50  # neutral default if insufficient data
            if len(candles_15m) >= 14:
                high, low, close = candles_15m['high'], candles_15m['low'], candles_15m['close']
                tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(
                    axis=1)
                atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

                dm_plus = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(
                    lower=0)
                dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(
                    lower=0)

                # atr is a scalar – handle zero
                if atr == 0:
                    atr = 1.0

                di_plus = 100 * dm_plus.ewm(alpha=1 / 14, adjust=False).mean() / atr
                di_minus = 100 * dm_minus.ewm(alpha=1 / 14, adjust=False).mean() / atr

                # denom is a Series – .replace() is safe
                denom = (di_plus + di_minus).replace(0, 1)
                dx = 100 * (di_plus - di_minus).abs() / denom
                adx_val = dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

                # RSI for logging (mirrors the internal calc composite_score() already does)
                delta = close.diff()
                gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
                rs = 999999 if loss == 0 else gain / loss
                rsi_val = 100 - (100 / (1 + rs))


            # HTF confirmation — must use futures token; index candles carry zero volume,
            # which makes calculate_vwap() return 0 and permanently fails this check.
            if not fut_instrument:
                current_signal = {"decision": "NO TRADE",
                                  "reason": "Futures unresolved — cannot compute HTF VWAP, failing closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('nifty_orderflow_signal', current_signal)
                return
            ht_candles = get_higher_tf_candles(fut_instrument['instrument_token'])

            # Compute the total volume for sanity check (used in both log and console)
            sanity_vol = ht_candles['volume'].sum() if not ht_candles.empty else 0

            logging.info(
                f"HTF sanity check: fut_instrument={fut_instrument['tradingsymbol'] if fut_instrument else None}, "
                f"ht_candles_volume_sum={sanity_vol}"
            )
            print(
                f"🔍 HTF sanity check: fut_instrument={fut_instrument['tradingsymbol']}, "
                f"ht_candles_volume_sum={sanity_vol}"
            )

            if ht_candles.empty or len(ht_candles) < 8:
                current_signal = {"decision": "NO TRADE", "reason": "HTF data unavailable — failing closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('nifty_orderflow_signal', current_signal)
                return

            today = now.date()
            ht_candles_today = ht_candles[pd.to_datetime(
                ht_candles['date']).dt.date == today] if 'date' in ht_candles.columns else ht_candles.tail(20)
            ht_vwap = calculate_vwap(ht_candles_today if not ht_candles_today.empty else ht_candles.tail(20))
            if ht_vwap <= 0:
                current_signal = {"decision": "NO TRADE", "reason": "HTF VWAP invalid — failing closed"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('nifty_orderflow_signal', current_signal)
                return

            ht_bias = "CALL" if ht_candles['close'].iloc[-1] > ht_vwap else "PUT"
            if ht_bias != bias:
                logging.info(f"🔴 HTF mismatch ({ht_bias} vs {bias}). Hard reject — no entry.")

                try:
                    htf_log_file = os.path.join(LOG_DIR, "nifty_htf_rejections.csv")
                    append_csv_row_safe(htf_log_file, {
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "spot_price": round(spot_ltp, 2),
                        "entry_tf_bias": bias,
                        "entry_tf_score": round(total_score, 1),
                        "htf_bias": ht_bias,
                        "htf_vwap": round(ht_vwap, 2),
                        "vix_value": round(vix_ltp, 2),
                        "market_regime": market_regime,
                        "outcome": "HARD_REJECTED",
                    })
                except Exception as e:
                    logging.warning(f"⚠️ Failed to log HTF rejection: {e}")

                current_signal = {
                    "decision": "NO TRADE",
                    "reason": f"HTF mismatch ({ht_bias} vs {bias}) — hard reject",
                    "last_scan": now.strftime("%H:%M:%S"),
                }
                safe_emit('nifty_orderflow_signal', current_signal)
                return


            # --- LOG: distance to opposing PDH/PDL (no gating) ---
            distance_to_level_atr = None
            if bias == "CALL" and key_levels.get("PDH") and entry_atr > 0:
                distance_to_level_atr = round((key_levels["PDH"] - spot_ltp) / entry_atr, 2)
            elif bias == "PUT" and key_levels.get("PDL") and entry_atr > 0:
                distance_to_level_atr = round((spot_ltp - key_levels["PDL"]) / entry_atr, 2)

            if total_score < 52 or bias == "NEUTRAL":
                print(f"🔴 REJECTED: Entry Score {round(total_score, 1)} < 52, Bias: {bias}")
                current_signal = {
                "decision": "NO TRADE",
                "reason": f"Entry Score {round(total_score, 1)}",
                "last_scan": now.strftime("%H:%M:%S"),
                "dte": dte,
                "expiry_date": expiry_date.isoformat(),
                "vix_value": round(vix_ltp, 2),
                "spot_price": round(spot_ltp, 2),
                "scenario": compute_scenario_probs(base_score, bias),
                "vix_regime": vix_regime,
                "vix_direction": vix_direction,
                "adx": round(adx_val, 2),
                "momentum_checks": 2 if adx_val > 25 else 1 if adx_val > 20 else 0,
                "kills": [],
                "failed_criteria": [],
                "time_elapsed": 0,
                "highest_premium": 0,
                "trail_stop": 0,
                "setup_quality": base_score,
                "signal_quality": signal_quality,
                "market_regime": market_regime,
                }
                safe_emit('nifty_orderflow_signal', current_signal)
                if active_trade:
                    monitor_signal = {
                        "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                        "is_monitoring": True,
                        "strike": active_trade.get('strike'),
                        "symbol": active_trade.get('symbol', ''),
                        "entry_ltp": round(entry_option_ltp, 2),
                        "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                        "bid_price": round(active_trade.get('trail_premium', active_trade.get('option_ltp', 0)), 2),
                        "option_sl": round(
                            active_trade.get('sl_price', max(entry_option_ltp * (1 - NIFTY_SL_PCT), 10.0)), 2),
                        "lots": active_trade.get('lots', 1),
                        "setup_quality": active_trade.get('setup_quality', 0),
                        "signal_quality": active_trade.get('signal_quality', 0),
                        "market_regime": active_trade.get('market_regime', ''),
                        "signal_id": active_trade.get('signal_id'),
                        "last_scan": now.strftime("%H:%M:%S"),
                        "primary_reason": f"[SIM] Monitoring (entry score {round(total_score, 1)})",
                        # Full dashboard fields
                        "dte": active_trade.get('dte', 0),
                        "vix_value": active_trade.get('vix_value', 0),
                        "spot_price": round(spot_ltp, 2),
                        "scenario": compute_scenario_probs(base_score, bias),
                        "vix_regime": vix_regime,
                        "vix_direction": vix_direction,
                        "adx": round(adx_val, 2),
                        "momentum_checks": 2 if adx_val > 25 else 1 if adx_val > 20 else 0,
                        "kills": [],
                        "failed_criteria": [],
                        "time_elapsed": round((now - trade_entry_time).total_seconds() / 60, 1) if trade_entry_time else 0,
                        "highest_premium": round(active_trade.get('highest_premium', entry_option_ltp), 2),
                        "trail_stop": round(
                            active_trade.get('highest_premium', entry_option_ltp) - active_trade.get('trail_distance', 0), 2)
                        if active_trade.get('trail_active', False) else None,
                        "trail_active": active_trade.get('trail_active', False),
                        "max_loss": round(
                            (entry_option_ltp - active_trade.get('sl_price',
                                                                 max(entry_option_ltp * (1 - NIFTY_SL_PCT),
                                                                     10.0))) * NIFTY_LOT_SIZE, 0),
                        "expiry_date": expiry_date.isoformat(),
                    }
                    safe_emit('nifty_orderflow_signal', monitor_signal)
                return

            rvol_score = feature_scores.get("rvol", {}).get("score", 0)
            oi_accel_score = feature_scores.get("oi_acceleration", {}).get("score", 0)
            if rvol_score <= 0 and oi_accel_score <= 0:
                current_signal = {
                    "decision": "NO TRADE",
                    "reason": f"No participation confirmation (RVOL {rvol_score}, OI accel {oi_accel_score})",
                    "last_scan": now.strftime("%H:%M:%S"),
                }
                safe_emit('nifty_orderflow_signal', current_signal)
                return

            atm = round(spot_ltp / 100) * 100
            if dte > 2:
                if vix_ltp <= 15:
                    offset = 100
                elif vix_ltp <= 20:
                    offset = 200
                elif vix_ltp <= 25:
                    offset = 300
                else:
                    offset = 400
                candidate_strike = atm + offset if bias == "CALL" else atm - offset
            else:
                candidate_strike = atm


            option_type = "CE" if bias == "CALL" else "PE"

            try:
                opt_row = chain_df[(chain_df['strike'] == candidate_strike) & (chain_df['instrument_type'] == option_type)]
                if opt_row.empty:
                    current_signal = {"decision": "NO TRADE", "reason": "Candidate strike not in chain"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('nifty_orderflow_signal', current_signal)
                    print(f"🔴 REJECTED: Strike {candidate_strike} not found in chain (bias: {bias})")
                    return
                option_symbol = opt_row.iloc[0]['tradingsymbol']
            except Exception as e:
                logging.warning(f"⚠️ NIFTY option symbol resolution failed | bias={bias} strike={candidate_strike} expiry={expiry_date}: {e}")
                current_signal = {"decision": "NO TRADE", "reason": f"Option symbol resolution failed: {type(e).__name__}"}
                current_signal["last_scan"] = now.strftime("%H:%M:%S")
                safe_emit('nifty_orderflow_signal', current_signal)
                return

            if active_trade is None:
                try:
                    opt_quote = kite_call_with_timeout(kite.quote, [f"NFO:{option_symbol}"])
                    if opt_quote is None:
                        current_signal = {"decision": "NO TRADE", "reason": "Option quote fetch failed"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        print(f"🔴 REJECTED: Option quote fetch failed for {option_symbol}")
                        return

                    option_ltp = opt_quote.get(f"NFO:{option_symbol}", {}).get('last_price', 0)

                    # --- Spread/Liquidity Filter — moved ahead of the premium floor so the
                    # floor check uses the real fillable price (ask), not stale LTP ---
                    depth = opt_quote.get(f"NFO:{option_symbol}", {}).get('depth', {})
                    bid = depth.get('buy', [{}])[0].get('price', 0)
                    ask = depth.get('sell', [{}])[0].get('price', 0)
                    if bid <= 0 or ask <= 0:
                        current_signal = {"decision": "NO TRADE", "reason": "No two-sided market"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        print(f"🔴 REJECTED: No two-sided market for {option_symbol}")
                        return

                    spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                    if spread_pct > MAX_SPREAD_PCT:
                        current_signal = {"decision": "NO TRADE", "reason": f"Spread too wide ({spread_pct:.1f}%)"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        print(f"🔴 REJECTED: Spread {spread_pct:.1f}% > {MAX_SPREAD_PCT}%")
                        return
                    # --- ADDED: simulate a realistic buy fill at the ask, not LTP ---
                    option_ltp = ask
                    if option_ltp <= 40:
                        current_signal = {"decision": "NO TRADE", "reason": f"Option premium too low ({option_ltp})"}
                        current_signal["last_scan"] = now.strftime("%H:%M:%S")
                        safe_emit('nifty_orderflow_signal', current_signal)
                        print(f"🔴 REJECTED: Option premium too low ({option_ltp} <= 40)")
                        return
                except Exception as e:
                    logging.warning(f"⚠️ NIFTY entry quote/spread check exception for {option_symbol}: {e}")
                    current_signal = {"decision": "NO TRADE",
                                      "reason": f"Exception fetching option quote: {type(e).__name__}"}
                    current_signal["last_scan"] = now.strftime("%H:%M:%S")
                    safe_emit('nifty_orderflow_signal', current_signal)
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
                    "symbol": option_symbol,
                    "lots": 1,
                    "strike": candidate_strike,
                    "setup_quality": base_score,
                    "signal_quality": signal_quality,
                    "dte": dte,
                    "dead_trade_minutes": NIFTY_DEAD_TRADE_MINUTES_ATM if dte <= 2 else NIFTY_DEAD_TRADE_MINUTES,
                    "vix_value": round(vix_ltp, 2),
                    "adx": round(adx_val, 2),
                    "rsi": round(rsi_val, 2),
                    "entry_spread_pct": round(spread_pct, 2),
                    "underlying": "NIFTY",
                    "underlying_ltp": spot_ltp,
                    "signal_id": signal_id,
                    "market_regime": market_regime,
                    "sl_price": max(option_ltp * (1 - NIFTY_SL_PCT), 10.0),
                    "feature_scores": convert_numpy({k: v['score'] for k, v in feature_scores.items()}),
                    "trail_distance": max(8, min(25, int(entry_atr * 0.25))),
                    "activation_threshold": max(NIFTY_TRAIL_ACTIVATION, max(8, min(25, int(entry_atr * 0.25)))),
                    "entry_atr": round(entry_atr, 2),
                    "distance_to_level_atr": distance_to_level_atr,
                    "last_quote_time": now,
                    "feature_snapshot": convert_numpy({k: v['value'] if not isinstance(v, dict) else v for k, v in feature_scores.items()})
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
                    "underlying_entry": spot_ltp,
                    "base_score": base_score,
                    "oi_class": comp.get("oi_class"),
                    "score_reasons": comp.get("reasons"),
                    "signal_quality": signal_quality,
                    "market_regime": market_regime,
                    "feature_scores": active_trade['feature_scores'],
                    "feature_snapshot": active_trade['feature_snapshot'],
                    "dte": dte,
                    "dead_trade_minutes": active_trade.get('dead_trade_minutes'),
                    "vix": vix_ltp,
                    "adx": active_trade.get('adx', 0),
                    "rsi": active_trade.get('rsi', 0),
                    "entry_spread_pct": active_trade.get('entry_spread_pct', 0),
                    "entry_atr": active_trade.get('entry_atr', 0),
                    "distance_to_level_atr": active_trade.get('distance_to_level_atr'),
                    "is_sim": True
                })
                logging.info(f"🔁 [SIM] ENTRY: {option_symbol} @ {option_ltp} | Base: {base_score} | Total: {signal_quality} | ID: {signal_id}")

            if active_trade:
                monitor_signal = {
                    "decision": f"{active_trade.get('bias', '')} BUY ([SIM])",
                    "is_monitoring": True,
                    "strike": active_trade.get('strike'),
                    "symbol": active_trade.get('symbol', ''),
                    "entry_ltp": round(entry_option_ltp, 2),
                    "option_ltp": round(active_trade.get('option_ltp', 0), 2),
                    "bid_price": round(active_trade.get('trail_premium', active_trade.get('option_ltp', 0)), 2),
                    "option_sl": round(active_trade.get('sl_price', max(entry_option_ltp * (1 - NIFTY_SL_PCT), 10.0)), 2),
                    "lots": active_trade.get('lots', 1),
                    "setup_quality": active_trade.get('setup_quality', 0),
                    "signal_quality": active_trade.get('signal_quality', 0),
                    "market_regime": active_trade.get('market_regime', ''),
                    "signal_id": active_trade.get('signal_id'),
                    "last_scan": now.strftime("%H:%M:%S"),
                    "primary_reason": f"[SIM] Score {active_trade.get('signal_quality', 0)}",
                    # Full dashboard fields
                    "dte": active_trade.get('dte', 0),
                    "vix_value": active_trade.get('vix_value', 0),
                    "spot_price": round(spot_ltp, 2),
                    "scenario": compute_scenario_probs(base_score, bias),
                    "vix_regime": vix_regime,
                    "vix_direction": vix_direction,
                    "adx": round(adx_val, 2),
                    "momentum_checks": 2 if adx_val > 25 else 1 if adx_val > 20 else 0,
                    "kills": [],
                    "failed_criteria": [],
                    "time_elapsed": round((now - trade_entry_time).total_seconds() / 60, 1) if trade_entry_time else 0,
                    "highest_premium": round(active_trade.get('highest_premium', entry_option_ltp), 2),
                    "trail_stop": round(
                        active_trade.get('highest_premium', entry_option_ltp) - active_trade.get('trail_distance', 0), 2)
                    if active_trade.get('trail_active', False) else None,
                    "trail_active": active_trade.get('trail_active', False),
                    "max_loss": round(
                        (entry_option_ltp - active_trade.get('sl_price',
                                                             max(entry_option_ltp * (1 - NIFTY_SL_PCT),
                                                                 10.0))) * NIFTY_LOT_SIZE, 0),
                    "expiry_date": expiry_date.isoformat(),
                }
                safe_emit('nifty_orderflow_signal', monitor_signal)

            # --- FALLBACK: emit NO TRADE if we reached the end without any trade ---
            if active_trade is None:
                current_signal = {
                    "decision": "NO TRADE",
                    "reason": "No Entry Signal Generated",
                    "last_scan": now.strftime("%H:%M:%S"),
                    "dte": dte,
                    "expiry_date": expiry_date.isoformat(),
                    "vix_value": round(vix_ltp, 2) if 'vix_ltp' in locals() else 0,
                    "spot_price": round(spot_ltp, 2),
                    "scenario": compute_scenario_probs(base_score, bias),
                    "vix_regime": vix_regime,
                    "vix_direction": vix_direction,
                    "adx": round(adx_val, 2),
                    "momentum_checks": 2 if adx_val > 25 else 1 if adx_val > 20 else 0,
                    "kills": [],
                    "failed_criteria": [],
                    "time_elapsed": 0,
                    "highest_premium": 0,
                    "trail_stop": 0,
                    "setup_quality": base_score,
                    "signal_quality": signal_quality,
                    "market_regime": market_regime,
                }
                safe_emit('nifty_orderflow_signal', current_signal)
        except Exception as e:
            logging.exception(f"Nifty order-flow scan error: {e}")
            print(f"❌ SCAN ERROR: {e}")

def background_scanner():
    while True:
        run_nifty_orderflow_scan()
        time.sleep(60 if active_trade is None else 1)

# ======================== FLASK ROUTES ========================
@app.route('/')
def index():
    response = send_from_directory('templates', 'orderflow_nifty.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@socketio.on('request_signal')
def handle_request():
    with state_lock:
        sig = current_signal.copy()
    safe_emit('nifty_orderflow_signal', sig)

@socketio.on('connect')
def handle_connect():
    print("✅ Client connected to Socket.IO")
    with state_lock:
        sig = current_signal.copy()
    emit('nifty_orderflow_signal', sig)

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
            safe_emit('nifty_orderflow_signal', current_signal)
            logging.info(f"Manual exit triggered - PnL: ₹{exit_pnl:.0f}")
        else:
            logging.info("Manual exit requested but no active trade")



if __name__ == "__main__":
    load_state()
    threading.Thread(target=background_scanner, daemon=True).start()
    print(f"🚀 Nifty Order-Flow Engine (SHADOW) v{STRATEGY_VERSION} running on http://localhost:{PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False)