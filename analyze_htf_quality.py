"""
HTF Veto Quality Analysis
==========================
Determines whether the HTF (1-hour VWAP) gate was right or wrong for each
rejection logged in *_htf_rejections.csv.

Usage:
    python analyze_htf_quality.py /path/to/logs /path/to/.env

Expects:
    - logs_dir containing crude_htf_rejections.csv and/or nifty_htf_rejections.csv
    - .env file with OF_API_KEY and OF_ACCESS_TOKEN (same as engines)

Output:
    - Prints summary: count of right vs wrong vetoes, with breakdown by regime.
"""

import os
import sys
import pandas as pd
import datetime
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

LOOKBACK_DAYS = 7  # how many days back to fetch candles (adjust as needed)
LAGS_MINUTES = [15, 30, 60]  # check price at these intervals after rejection

# Instrument tokens (these match your engines; they should be correct)
# If you want to use spot for Nifty, you can adjust.
INSTRUMENT_TOKENS = {
    "crude": None,   # will be resolved dynamically from the engine's logic
    "nifty": None,   # will be resolved dynamically from the engine's logic
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_rejections(logs_dir, instrument):
    path = os.path.join(logs_dir, f"{instrument}_htf_rejections.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_token_for_instrument(kite, instrument):
    """
    Resolve the correct token for Crude futures and Nifty spot/futures.
    This mirrors your engine's expiry/futures resolution.
    For simplicity, you can also pass the token as an argument if you know it.
    """
    if instrument == "crude":
        # Use the same logic as resolve_current_expiry() in crude_orderflow.py
        # But we'll just fetch the current expiry from the instrument list.
        instruments = kite.instruments("MCX")
        futs = [i for i in instruments if i['name'] == 'CRUDEOIL' and i['instrument_type'] == 'FUT']
        futs.sort(key=lambda x: x['expiry'])
        today = datetime.datetime.now().date()
        for f in futs:
            exp = f['expiry']
            if isinstance(exp, str):
                exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
            else:
                exp_date = exp
            if exp_date > today:
                return f['instrument_token']
        return None
    elif instrument == "nifty":
        # For Nifty, we use the NIFTY 50 spot token (256265) - same as your engine.
        return 256265
    return None

def fetch_candles_for_day(kite, token, date):
    """Fetch 5-minute candles for a specific date."""
    from_date = date.strftime("%Y-%m-%d")
    to_date = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        data = kite.historical_data(token, from_date, to_date, "5minute")
        if data is None:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"  ⚠️ Error fetching candles for {date}: {e}")
        return pd.DataFrame()

def get_price_later(df_candles, ref_time, lag_minutes):
    """Get the close price exactly `lag_minutes` after ref_time, or the next available."""
    target = ref_time + datetime.timedelta(minutes=lag_minutes)
    # find the first candle with timestamp >= target
    candidates = df_candles[df_candles["timestamp"] >= target]
    if candidates.empty:
        return None
    return candidates.iloc[0]["close"]

# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------

def get_token_for_date(kite, instrument, target_date, futs_cache={}):
    """Resolve the correct instrument token as of a specific historical date,
        not the current date — matters for Crude since futures roll monthly."""
    if instrument == "nifty":
        return 256265  # spot token never changes, no rollover risk

    if "crude_futs" not in futs_cache:
        instruments = kite.instruments("MCX")
        futs_cache["crude_futs"] = [i for i in instruments if i['name'] == 'CRUDEOIL' and i['instrument_type'] == 'FUT']
        futs_cache["crude_futs"].sort(key=lambda x: x['expiry'])

    for f in futs_cache["crude_futs"]:
        exp = f['expiry']
        exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date() if isinstance(exp, str) else exp
        # the contract that was front-month on target_date is the one whose expiry is the
        # nearest one still in the future relative to that date
        if exp_date > target_date:
            return f['instrument_token']
    return None


def analyze_instrument(logs_dir, instrument, kite):
    print(f"\n--- {instrument.upper()} HTF VETO QUALITY ---")
    df_rej = load_rejections(logs_dir, instrument)
    if df_rej is None:
        print("  No HTF rejections file found.")
        return

    print(f"  Loaded {len(df_rej)} rejections.")

    # Group by date, resolving the correct contract token PER DATE
    dates = df_rej["timestamp"].dt.date.unique()
    candle_cache = {}
    for d in dates:
        token = get_token_for_date(kite, instrument, d)
        if token is None:
            print(f"  ⚠️ Could not resolve token for {d}. Skipping.")
            continue
        print(f"  Fetching 5m candles for {d} (token {token})...")
        df_candles = fetch_candles_for_day(kite, token, d)
        if not df_candles.empty:
            candle_cache[d] = df_candles
        else:
            print(f"  ⚠️ No candles for {d}")

    # Evaluate each rejection
    results = []
    for idx, row in df_rej.iterrows():
        ts = row["timestamp"]
        date = ts.date()
        if date not in candle_cache:
            continue
        df_candles = candle_cache[date]
        entry_bias = row["entry_tf_bias"]
        # Price at rejection
        price_at_rej = row["futures_price"] if "futures_price" in row else row.get("spot_price", None)
        if price_at_rej is None:
            continue

        # For each lag, get price later and determine if veto was right
        for lag in LAGS_MINUTES:
            price_later = get_price_later(df_candles, ts, lag)
            if price_later is None:
                continue
            # Determine if price moved in the direction of the entry bias
            if entry_bias == "CALL":
                # CALL: price should rise
                veto_right = price_later < price_at_rej  # price fell → veto saved us
            elif entry_bias == "PUT":
                veto_right = price_later > price_at_rej  # price rose → veto saved us
            else:
                continue

            results.append({
                "timestamp": ts,
                "entry_bias": entry_bias,
                "htf_bias": row["htf_bias"],
                "lag_minutes": lag,
                "price_at_rej": price_at_rej,
                "price_later": price_later,
                "veto_right": veto_right,
                "regime": row.get("market_regime", "UNKNOWN"),
                "score": row.get("entry_tf_score", None),
            })

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("  No results computed (maybe no candles for the rejection dates?).")
        return

    # Summary per lag
    for lag in LAGS_MINUTES:
        subset = df_results[df_results["lag_minutes"] == lag]
        if subset.empty:
            continue
        right = subset["veto_right"].sum()
        total = len(subset)
        pct_right = right / total * 100
        print(f"\n  At {lag} minutes later: {right}/{total} ({pct_right:.1f}%) vetoes were RIGHT.")
        if total > 10:
            if pct_right > 70:
                print("    → HTF gate is clearly helping (saving >70% of rejected trades).")
            elif pct_right < 40:
                print("    → HTF gate is hurting (saving <40% of rejected trades). Consider loosening or removing it.")
            else:
                print("    → Mixed results. Could be dependent on regime; check breakdown below.")

    # Breakdown by market regime (optional)
    if "regime" in df_results.columns and len(df_results["regime"].unique()) > 1:
        print("\n  Breakdown by market regime (30-min lag):")
        subset = df_results[df_results["lag_minutes"] == 30]
        for regime, group in subset.groupby("regime"):
            right = group["veto_right"].sum()
            total = len(group)
            if total > 0:
                print(f"    {regime}: {right}/{total} ({right/total*100:.1f}%) right")

    # Also count total trades (if any) – just for context
    print("\n  HTF veto count:", len(df_rej))
    print("  ⚠️ To answer 'would the 5m signal have won?', compare price at 30min later.")
    print("  If price moved in the entry bias direction, the veto was wrong.")
    print("  If it moved against, the veto was right.")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_htf_quality.py /path/to/logs")
        sys.exit(1)

    logs_dir = sys.argv[1]
    # Load environment for Kite
    load_dotenv(".env.orderflow")  # adjust if your .env file is elsewhere
    api_key = os.getenv("OF_API_KEY")
    access_token = os.getenv("OF_ACCESS_TOKEN")
    if not api_key or not access_token:
        print("❌ OF_API_KEY or OF_ACCESS_TOKEN not found in .env.orderflow")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Analyze both instruments if files exist
    instruments = ["crude", "nifty"]
    for inst in instruments:
        # Check if rejection file exists
        if os.path.exists(os.path.join(logs_dir, f"{inst}_htf_rejections.csv")):
            analyze_instrument(logs_dir, inst, kite)
        else:
            print(f"⚠️ No rejection file for {inst}, skipping.")