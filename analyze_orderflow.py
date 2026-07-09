"""
Orderflow Shadow-Mode Analysis Script
=====================================
Analyzes logs produced by crude_orderflow.py / nifty_orderflow.py to answer
the 12 blueprint questions agreed on before data collection began.

USAGE:
    python analyze_orderflow.py /path/to/logs

Expects the standard log filenames the engines already write:
    {instrument}_orderflow_sim_summary.csv
    {instrument}_regime_performance.csv
    {instrument}_htf_rejections.csv
    {instrument}_orderflow.log

Where {instrument} is "nifty" or "crude". Missing files are skipped with a
note rather than crashing the whole report — you don't need every file
present to get partial results.
"""

import os
import re
import json
import sys
import pandas as pd
import numpy as np

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


# ------------------------------------------------------------------
# Loaders — every one is defensive; missing/malformed files degrade
# gracefully instead of crashing the whole report.
# ------------------------------------------------------------------

def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  ⚠️ Could not read {path}: {e}")
        return None


def load_sim_summary(logs_dir, instrument):
    path = os.path.join(logs_dir, f"{instrument}_orderflow_sim_summary.csv")
    df = load_csv_safe(path)
    if df is None:
        return None
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_regime_performance(logs_dir, instrument):
    return load_csv_safe(os.path.join(logs_dir, f"{instrument}_regime_performance.csv"))


def load_htf_rejections(logs_dir, instrument):
    path = os.path.join(logs_dir, f"{instrument}_htf_rejections.csv")
    df = load_csv_safe(path)
    if df is None:
        return None
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


EMIT_RE = re.compile(
    r"^(?P<ts>[\d\-]+ [\d:,]+) - (?P<level>\w+) - EMIT \w+: (?P<decision>[^|]+) \| (?P<reason>.*)$"
)


def parse_log_rejections(logs_dir, instrument):
    path = os.path.join(logs_dir, f"{instrument}_orderflow.log")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = EMIT_RE.match(line.strip())
            if not m:
                continue
            rows.append({
                "timestamp": m.group("ts").split(",")[0],
                "decision": m.group("decision").strip(),
                "reason": m.group("reason").strip(),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["reason_class"] = df["reason"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    return df


def safe_json_col(series):
    """Parse a column of JSON strings (e.g. feature_scores) into dicts, skipping bad rows."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append({})
            continue
        try:
            out.append(json.loads(v) if isinstance(v, str) else {})
        except Exception:
            out.append({})
    return out


# ------------------------------------------------------------------
# Section 1 — Baseline performance / expectancy
# ------------------------------------------------------------------

def section_baseline(df):
    print("\n### 1. BASELINE PERFORMANCE")
    if df is None:
        print("  No sim_summary data yet.")
        return
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / n if n else 0
    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    print(f"  Trades: {n} | Win rate: {win_rate:.1%}")
    print(f"  Avg win: ₹{avg_win:.0f} | Avg loss: ₹{avg_loss:.0f}")
    print(f"  Expectancy per trade: ₹{expectancy:.1f}")
    if len(wins):
        print(f"  Largest win: ₹{wins['pnl'].max():.0f}")
    if len(losses):
        print(f"  Largest loss: ₹{losses['pnl'].min():.0f}")
    if n < 30:
        print(f"  ⚠️ Sample size ({n}) is small — treat these numbers as a first read, not a verdict.")


# ------------------------------------------------------------------
# Section 2 — HTF rejection quality (self-contained parts only;
# "did price actually reverse" needs supplementary price history —
# flagged, not faked)
# ------------------------------------------------------------------

def section_htf(df_rej):
    print("\n### 2. HTF FILTER QUALITY")
    if df_rej is None:
        print("  No HTF rejections logged yet.")
        return
    print(f"  Total HTF-vetoed signals: {len(df_rej)}")
    if "entry_tf_bias" in df_rej.columns:
        print("  Vetoed bias breakdown:\n" + df_rej["entry_tf_bias"].value_counts().to_string())
    if "market_regime" in df_rej.columns:
        print("\n  By regime at time of veto:\n" + df_rej["market_regime"].value_counts().to_string())
    if "entry_tf_score" in df_rej.columns:
        print(f"\n  Avg score of vetoed signals: {df_rej['entry_tf_score'].mean():.1f}")
    print("\n  NOTE: to check whether the HTF veto was actually *right* (price reversed)")
    print("  or *wrong* (5m signal would've won), this script needs the underlying")
    print("  spot/futures price a few candles after each rejection timestamp — that data")
    print("  isn't in htf_rejections.csv itself. If you export 1m/5m candles covering")
    print("  the session, I can join them against these timestamps for the real verdict.")


# ------------------------------------------------------------------
# Section 3 — PDH/PDL proximity outcome
# ------------------------------------------------------------------

def section_distance_to_level(df):
    print("\n### 3. PDH/PDL PROXIMITY (distance_to_level_atr)")
    if df is None or "distance_to_level_atr" not in df.columns:
        print("  Column not present or no data yet.")
        return
    d = df.dropna(subset=["distance_to_level_atr"])
    if d.empty:
        print("  No trades had a nearby opposing level logged (all were far from PDH/PDL, or trades not yet closed).")
        return
    close = d[d["distance_to_level_atr"] < 0.5]
    far = d[d["distance_to_level_atr"] >= 0.5]
    for label, sub in (("Close to level (<0.5 ATR)", close), ("Far from level (>=0.5 ATR)", far)):
        if len(sub):
            wr = (sub["pnl"] > 0).mean()
            print(f"  {label}: n={len(sub)}, win rate={wr:.1%}, avg PnL=₹{sub['pnl'].mean():.0f}")
        else:
            print(f"  {label}: n=0")
    if len(close) and len(far):
        verdict = "would likely have HELPED" if close["pnl"].mean() < far["pnl"].mean() else "would likely have COST you trades"
        print(f"  → A 0.5xATR proximity gate {verdict}, based on this sample.")


# ------------------------------------------------------------------
# Section 4 — Rejection reason tally + signal frequency
# ------------------------------------------------------------------

def section_rejection_tally(df_log):
    print("\n### 4. REJECTION REASON TALLY / SIGNAL FREQUENCY")
    if df_log is None:
        print("  No parsable EMIT lines found in the log file yet.")
        return
    counts = df_log["reason_class"].value_counts()
    print(counts.to_string())
    total = len(df_log)
    days = df_log["timestamp"].dt.date.nunique() if "timestamp" in df_log.columns else 1
    print(f"\n  Total scans logged: {total} across {days} day(s) (~{total/max(days,1):.0f}/day)")
    if "timestamp" in df_log.columns:
        by_hour = df_log.assign(hour=df_log["timestamp"].dt.hour)["hour"].value_counts().sort_index()
        print("\n  Activity by hour of day:\n" + by_hour.to_string())


# ------------------------------------------------------------------
# Section 5 — Regime performance
# ------------------------------------------------------------------

def section_regime(df):
    print("\n### 5. REGIME PERFORMANCE")
    if df is None:
        print("  No regime_performance data yet.")
        return
    print(df.sort_values("avg_pnl", ascending=False).to_string(index=False))
    negative = df[df["avg_pnl"] < 0]
    if len(negative):
        print(f"\n  ⚠️ Regimes with negative avg PnL so far: {', '.join(negative['regime'].tolist())}")
        print("  (Not enough to act on yet at low sample size — just noting it.)")


# ------------------------------------------------------------------
# Section 6 — DTE / expiry effects
# ------------------------------------------------------------------

def section_dte(df):
    print("\n### 6. DTE / EXPIRY EFFECTS")
    if df is None or "dte" not in df.columns:
        print("  No data yet.")
        return
    g = df.groupby("dte")["pnl"].agg(["count", "mean"])
    print(g.to_string())


# ------------------------------------------------------------------
# Section 7 — Fill realism sanity check
# ------------------------------------------------------------------

def section_fill_realism(df):
    print("\n### 7. FILL REALISM SANITY CHECK")
    if df is None:
        print("  No data yet.")
        return
    print(f"  Avg entry price: {df['entry_price'].mean():.2f} | Avg exit price: {df['exit_price'].mean():.2f}")
    print("  (No independent bid/ask history logged per-trade to compare against —")
    print("   this just confirms trades are populating with sane, non-zero prices.)")


# ------------------------------------------------------------------
# Section 8 — Score calibration & feature contribution
# ------------------------------------------------------------------

def section_score_calibration(df):
    print("\n### 8. SCORE CALIBRATION & FEATURE CONTRIBUTION")
    if df is None or "signal_quality" not in df.columns:
        print("  No data yet.")
        return
    wins = df[df["pnl"] > 0]["signal_quality"]
    losses = df[df["pnl"] <= 0]["signal_quality"]
    if len(wins) and len(losses):
        print(f"  Mean signal_quality — winners: {wins.mean():.1f} | losers: {losses.mean():.1f}")
        if wins.mean() - losses.mean() > 3:
            print("  → Score threshold appears to carry real signal (winners score meaningfully higher).")
        else:
            print("  → Score barely distinguishes winners from losers in this sample — worth more data before trusting it.")

    if "feature_scores" in df.columns:
        parsed = safe_json_col(df["feature_scores"])
        feat_df = pd.DataFrame(parsed)
        if not feat_df.empty:
            feat_df["pnl"] = df["pnl"].values
            print("\n  Feature value correlation with PnL:")
            corrs = {}
            for col in feat_df.columns:
                if col == "pnl":
                    continue
                try:
                    numeric = pd.to_numeric(feat_df[col], errors="coerce")
                    if numeric.notna().sum() >= 5 and numeric.std() > 0:
                        corrs[col] = numeric.corr(feat_df["pnl"])
                except Exception:
                    continue
            if corrs:
                for k, v in sorted(corrs.items(), key=lambda x: -abs(x[1])):
                    print(f"    {k}: {v:+.2f}")
            else:
                print("    Not enough variation yet to compute correlations.")


# ------------------------------------------------------------------
# Section 9 — Exit efficiency (capture ratio)
# ------------------------------------------------------------------

def section_exit_efficiency(df):
    print("\n### 9. EXIT EFFICIENCY (CAPTURE RATIO)")
    if df is None or "mfe_pts" not in df.columns:
        print("  No data yet.")
        return
    d = df.copy()
    d["realized_pts"] = d["exit_price"] - d["entry_price"]
    d = d[d["mfe_pts"] > 0]
    if d.empty:
        print("  No trades with positive MFE yet.")
        return
    d["capture_ratio"] = d["realized_pts"] / d["mfe_pts"]
    avg_cr = d["capture_ratio"].mean()
    print(f"  Avg capture ratio: {avg_cr:.2f}")
    if avg_cr < 0.4:
        print("  → Giving back >60% of max favorable move — trail may be too loose / ratchet too slow.")
    elif avg_cr > 0.9:
        print("  → Capturing almost all of MFE — trail may be too tight, risking premature stop-outs on normal pullbacks.")
    else:
        print("  → Capture ratio in a reasonable middle range.")


# ------------------------------------------------------------------
# Section 10 — Holding period / theta decay
# ------------------------------------------------------------------

def section_holding_period(df):
    print("\n### 10. HOLDING PERIOD / THETA DECAY")
    if df is None or "holding_minutes" not in df.columns:
        print("  No data yet.")
        return
    bins = [0, 15, 30, 60, 90, 120, 1e9]
    labels = ["0-15m", "15-30m", "30-60m", "60-90m", "90-120m", "120m+"]
    d = df.copy()
    d["bucket"] = pd.cut(d["holding_minutes"], bins=bins, labels=labels)
    g = d.groupby("bucket", observed=True)["pnl"].agg(["count", "mean"])
    print(g.to_string())


# ------------------------------------------------------------------
# Section 11 — Strike selection validation (OTM vs ATM by DTE)
# ------------------------------------------------------------------

def section_strike_selection(df):
    print("\n### 11. STRIKE SELECTION VALIDATION (OTM vs ATM by DTE)")
    if df is None or "dte" not in df.columns:
        print("  No data yet.")
        return
    d = df.copy()
    d["strike_mode"] = np.where(d["dte"] > 2, "OTM (DTE>2)", "ATM (DTE<=2)")
    g = d.groupby("strike_mode")["pnl"].agg(["count", "mean", lambda x: (x > 0).mean()])
    g.columns = ["count", "avg_pnl", "win_rate"]
    print(g.to_string())


# ------------------------------------------------------------------
# Section 12 — Morning window theory (9:15-12:00 vs afternoon)
# ------------------------------------------------------------------

def section_morning_window(df):
    print("\n### 12. MORNING WINDOW THEORY (9:15-12:00 vs afternoon)")
    if df is None or "entry_time" not in df.columns:
        print("  No data yet.")
        return
    d = df.dropna(subset=["entry_time"]).copy()
    d["hour"] = d["entry_time"].dt.hour + d["entry_time"].dt.minute / 60
    morning = d[(d["hour"] >= 9.25) & (d["hour"] < 12.0)]
    afternoon = d[d["hour"] >= 12.0]
    for label, sub in (("Morning (9:15-12:00)", morning), ("Afternoon (12:00+)", afternoon)):
        if len(sub):
            wr = (sub["pnl"] > 0).mean()
            print(f"  {label}: n={len(sub)}, win rate={wr:.1%}, avg PnL=₹{sub['pnl'].mean():.0f}")
        else:
            print(f"  {label}: n=0")


# ------------------------------------------------------------------
# SECTION 13 (NEW) — Summary / Actionable Takeaways
# ------------------------------------------------------------------

def section_summary(sim, htf_rej, log_rej, instrument):
    print("\n### 13. SUMMARY / ACTIONABLE TAKEAWAYS")
    print("  (Use this as your decision checklist. Do not react to small samples.)")

    if sim is not None and len(sim):
        n = len(sim)
        win_rate = (sim["pnl"] > 0).mean()
        avg_pnl = sim["pnl"].mean()
        expectancy = (sim["pnl"].mean())
        print(f"\n  ✅ {instrument.upper()}: {n} closed trades")
        print(f"     Win rate: {win_rate:.1%} | Avg PnL: ₹{avg_pnl:.0f} | Expectancy: ₹{expectancy:.1f}")

        if n >= 30:
            if win_rate < 0.45:
                print("     → Win rate <45% – consider reviewing entry filters (HTF, PDH proximity, score).")
            elif win_rate > 0.6:
                print("     → Win rate >60% – good; now check avg R (reward/risk) to ensure it's not just tight stops.")
            if avg_pnl < 0:
                print("     → Negative expectancy overall – if this persists past 50 trades, revisit the scoring system.")
            else:
                print("     → Positive expectancy – continue collecting data to confirm stability.")
        else:
            print("     → Sample too small (<30) – continue collecting data before making any changes.")
    else:
        print(f"\n  ⚠️ {instrument.upper()}: no closed trades yet.")

    if htf_rej is not None and len(htf_rej):
        print(f"\n     HTF vetoed {len(htf_rej)} signals. This is the single biggest rejection reason to monitor.")
        if len(htf_rej) > 50 and (sim is None or len(sim) < 20):
            print("     → HTF is vetoing many signals but few trades are happening. Consider whether the gate is too strict.")
        else:
            print("     → HTF veto volume is within a reasonable range relative to trade count.")

    if log_rej is not None and not log_rej.empty:
        top_reason = log_rej["reason_class"].value_counts().index[0] if len(log_rej) else None
        if top_reason:
            print(f"\n     Top rejection reason: '{top_reason}'")
            if top_reason.lower().startswith("entry score"):
                if sim is not None and len(sim) >= 30:
                    print("     → Score is the dominant filter – watch if raising/lowering it changes trade quality.")
                else:
                    print(
                        "     → Score is the dominant filter, but sample is still small – note this, don't act on it yet.")
            elif top_reason.lower().startswith("htf"):
                print("     → HTF is the dominant filter – consider logging its quality (see Section 2).")

    print("\n  ✅ If you have 30+ trades, the numbers above give you a real signal to act on.")
    print("     If <30 trades, continue collecting – small samples lie.")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_report(logs_dir, instrument):
    print(f"\n{'='*70}\n{instrument.upper()} — ORDERFLOW ANALYSIS REPORT\n{'='*70}")
    sim = load_sim_summary(logs_dir, instrument)
    regime = load_regime_performance(logs_dir, instrument)
    htf_rej = load_htf_rejections(logs_dir, instrument)
    log_rej = parse_log_rejections(logs_dir, instrument)

    section_baseline(sim)
    section_htf(htf_rej)
    section_distance_to_level(sim)
    section_rejection_tally(log_rej)
    section_regime(regime)
    section_dte(sim)
    section_fill_realism(sim)
    section_score_calibration(sim)
    section_exit_efficiency(sim)
    section_holding_period(sim)
    section_strike_selection(sim)
    section_morning_window(sim)

    # --- NEW: Summary section at the end ---
    section_summary(sim, htf_rej, log_rej, instrument)


if __name__ == "__main__":
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for instrument in ("nifty", "crude"):
        run_report(logs_dir, instrument)