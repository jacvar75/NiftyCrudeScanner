#!/usr/bin/env python3
"""
NIFTY Order-Flow Log Analyzer

Parses nifty_orderflow.jsonl and prints a daily trade summary.
Usage:
    python nifty_log_analyzer.py                    # today's trades
    python nifty_log_analyzer.py 2026-07-20        # specific date
    python nifty_log_analyzer.py --all             # summary for all days

The log file is expected at: logs/nifty_orderflow.jsonl (relative to cwd).
"""

import json
import sys
import datetime
import os
from collections import defaultdict
from tabulate import tabulate  # optional – install via pip if desired

LOG_FILE = "logs/nifty_orderflow.jsonl"
DATE_FORMAT = "%Y-%m-%d"

def parse_iso(dt_str):
    """Parse ISO datetime string to datetime object."""
    return datetime.datetime.fromisoformat(dt_str)

def load_events():
    """Load all events from the JSONL log file."""
    events = []
    if not os.path.exists(LOG_FILE):
        print(f"❌ Log file not found: {LOG_FILE}")
        sys.exit(1)
    with open(LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events

def get_trades_for_date(events, target_date):
    """
    Return list of TRADE_CLOSED events whose exit_time is on target_date.
    Also includes the corresponding TRADE_OPENED data (if needed).
    """
    closed = []
    for ev in events:
        if ev.get('event') != 'TRADE_CLOSED':
            continue
        exit_time = ev.get('exit_time')
        if not exit_time:
            continue
        dt = parse_iso(exit_time)
        if dt.date() == target_date:
            closed.append(ev)
    return closed

def print_trade_table(trades):
    """Pretty print a list of trade dicts as a table."""
    if not trades:
        print("No trades found for this date.")
        return

    headers = [
        "Time", "Bias", "Strike", "Entry", "Exit", "PnL (₹)",
        "MFE", "MAE", "Holding (min)", "Exit Reason", "Regime", "SQ"
    ]
    rows = []
    for t in sorted(trades, key=lambda x: x.get('exit_time', '')):
        exit_dt = parse_iso(t['exit_time'])
        time_str = exit_dt.strftime("%H:%M:%S")
        rows.append([
            time_str,
            t.get('bias', ''),
            t.get('strike', ''),
            round(t.get('entry_price', 0), 2),
            round(t.get('exit_price', 0), 2),
            round(t.get('pnl', 0), 2),
            round(t.get('mfe_pts', 0), 2),
            round(t.get('mae_pts', 0), 2),
            round(t.get('holding_minutes', 0), 1),
            t.get('exit_reason', ''),
            t.get('market_regime', ''),
            t.get('signal_quality', 0)
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f"))

def daily_summary(trades):
    """Compute and print summary statistics."""
    if not trades:
        return
    total = len(trades)
    wins = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) < 0]
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    win_rate = len(wins) / total * 100 if total else 0
    avg_win = sum(t.get('pnl', 0) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.get('pnl', 0) for t in losses) / len(losses) if losses else 0
    max_win = max((t.get('pnl', 0) for t in trades), default=0)
    max_loss = min((t.get('pnl', 0) for t in trades), default=0)

    print("\n📊 DAILY SUMMARY")
    print("=" * 50)
    print(f"Total Trades       : {total}")
    print(f"Winning Trades     : {len(wins)}")
    print(f"Losing Trades      : {len(losses)}")
    print(f"Win Rate           : {win_rate:.1f}%")
    print(f"Total PnL          : ₹{total_pnl:,.2f}")
    print(f"Average Win        : ₹{avg_win:,.2f}")
    print(f"Average Loss       : ₹{avg_loss:,.2f}")
    print(f"Max Win            : ₹{max_win:,.2f}")
    print(f"Max Loss           : ₹{max_loss:,.2f}")
    print("=" * 50)

def analyze_all_days(events):
    """Group trades by date and print summary for each day."""
    by_date = defaultdict(list)
    for ev in events:
        if ev.get('event') != 'TRADE_CLOSED':
            continue
        exit_time = ev.get('exit_time')
        if not exit_time:
            continue
        dt = parse_iso(exit_time).date()
        by_date[dt].append(ev)

    if not by_date:
        print("No trades found.")
        return

    for dt in sorted(by_date.keys()):
        print(f"\n{'='*60}")
        print(f"📅 {dt.strftime('%A, %B %d, %Y')}")
        trades = by_date[dt]
        print_trade_table(trades)
        daily_summary(trades)

def main():
    events = load_events()
    if not events:
        print("No events loaded from log file.")
        return

    # Determine what to do based on arguments
    if "--all" in sys.argv:
        analyze_all_days(events)
        return

    # Parse target date
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        try:
            target_date = datetime.datetime.strptime(sys.argv[1], DATE_FORMAT).date()
        except ValueError:
            print(f"Invalid date format. Use YYYY-MM-DD (e.g., 2026-07-20)")
            sys.exit(1)
    else:
        target_date = datetime.datetime.now().date()

    trades = get_trades_for_date(events, target_date)
    print(f"\n📅 Trades for {target_date.strftime('%A, %B %d, %Y')}")
    print("=" * 60)
    print_trade_table(trades)
    daily_summary(trades)

if __name__ == "__main__":
    main()