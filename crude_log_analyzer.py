#!/usr/bin/env python3
"""
CRUDE Order-Flow Log Analyzer

Parses crude_orderflow.jsonl and prints a detailed daily trade summary.
Usage:
    python crude_log_analyzer.py                    # today's trades + month-to-date
    python crude_log_analyzer.py 2026-07-20        # specific date + month-to-date
    python crude_log_analyzer.py --all             # summary for all days

The log file is expected at: logs/crude_orderflow.jsonl (relative to cwd).
"""

import json
import sys
import datetime
import os
from collections import defaultdict
from tabulate import tabulate  # optional – install via pip if desired

LOG_FILE = "logs/crude_orderflow.jsonl"
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

def get_trades_for_month(events, target_date):
    """Return all trades from the 1st of target_date's month up to target_date (inclusive)."""
    month_start = target_date.replace(day=1)
    closed = []
    for ev in events:
        if ev.get('event') != 'TRADE_CLOSED':
            continue
        exit_time = ev.get('exit_time')
        if not exit_time:
            continue
        dt = parse_iso(exit_time).date()
        if month_start <= dt <= target_date:
            closed.append(ev)
    return closed

def print_trade_table(trades):
    """Pretty print a list of trade dicts as a table."""
    if not trades:
        print("No trades found for this date.")
        return

    headers = [
        "Time", "Bias", "Strike", "DTE", "Entry", "Exit", "PnL (₹)",
        "MFE", "MAE", "Holding (min)", "Deadline (min)", "Exit Reason", "Regime", "SQ"
    ]
    rows = []
    for t in sorted(trades, key=lambda x: x.get('exit_time', '')):
        exit_dt = parse_iso(t['exit_time'])
        time_str = exit_dt.strftime("%H:%M:%S")
        # Try to get cutoff from either field name (crude uses dead_trade_cutoff_minutes, nifty uses dead_trade_minutes)
        cutoff = t.get('dead_trade_cutoff_minutes') or t.get('dead_trade_minutes', '-')
        rows.append([
            time_str,
            t.get('bias', ''),
            t.get('strike', ''),
            t.get('dte', ''),
            round(t.get('entry_price', 0), 2),
            round(t.get('exit_price', 0), 2),
            round(t.get('pnl', 0), 2),
            round(t.get('mfe_pts', 0), 2),
            round(t.get('mae_pts', 0), 2),
            round(t.get('holding_minutes', 0), 1),
            cutoff,
            t.get('exit_reason', ''),
            t.get('market_regime', ''),
            t.get('signal_quality', 0)
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f"))

def print_dte_breakdown(trades):
    """Print a breakdown of performance by DTE."""
    if not trades:
        return

    dte_groups = defaultdict(list)
    for t in trades:
        dte = t.get('dte', 'N/A')
        dte_groups[dte].append(t)

    if len(dte_groups) <= 1:
        return

    print("\n📊 DTE BREAKDOWN")
    print("=" * 80)
    headers = ["DTE", "Trades", "Wins", "Losses", "Win Rate", "Total PnL", "Avg Trade", "Profit Factor"]
    rows = []
    for dte in sorted([k for k in dte_groups.keys() if k != 'N/A']):
        trades_dte = dte_groups[dte]
        total = len(trades_dte)
        wins = [t for t in trades_dte if t.get('pnl', 0) > 0]
        losses = [t for t in trades_dte if t.get('pnl', 0) < 0]
        total_pnl = sum(t.get('pnl', 0) for t in trades_dte)
        win_rate = len(wins) / total * 100 if total else 0
        avg_trade = total_pnl / total if total else 0
        gross_profit = sum(t.get('pnl', 0) for t in wins)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        rows.append([
            dte,
            total,
            len(wins),
            len(losses),
            f"{win_rate:.1f}%",
            f"₹{total_pnl:,.2f}",
            f"₹{avg_trade:,.2f}",
            f"{profit_factor:.2f}"
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 80)

def print_summary(trades, title):
    """Print a detailed summary (used for both daily and month-to-date)."""
    if not trades:
        print(f"\n{title} – No trades found.")
        return

    total = len(trades)
    wins = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) < 0]
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    win_rate = len(wins) / total * 100 if total else 0

    gross_profit = sum(t.get('pnl', 0) for t in wins)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    avg_trade = total_pnl / total if total else 0

    avg_win = sum(t.get('pnl', 0) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.get('pnl', 0) for t in losses) / len(losses) if losses else 0
    max_win = max((t.get('pnl', 0) for t in trades), default=0)
    max_loss = min((t.get('pnl', 0) for t in trades), default=0)

    print(f"\n{title}")
    print("=" * 50)
    print(f"Total Trades       : {total}")
    print(f"Winning Trades     : {len(wins)}")
    print(f"Losing Trades      : {len(losses)}")
    print(f"Win Rate           : {win_rate:.1f}%")
    print(f"Total PnL          : ₹{total_pnl:,.2f}")
    print(f"Average Trade      : ₹{avg_trade:,.2f}")
    print(f"Gross Profit       : ₹{gross_profit:,.2f}")
    print(f"Gross Loss         : ₹{gross_loss:,.2f}")
    print(f"Profit Factor      : {profit_factor:.2f}")
    print(f"Average Win        : ₹{avg_win:,.2f}")
    print(f"Average Loss       : ₹{avg_loss:,.2f}")
    print(f"Max Win            : ₹{max_win:,.2f}")
    print(f"Max Loss           : ₹{max_loss:,.2f}")
    print("=" * 50)

    print_dte_breakdown(trades)

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
        print_summary(trades, f"DAILY SUMMARY – {dt.strftime('%Y-%m-%d')}")

def main():
    events = load_events()
    if not events:
        print("No events loaded from log file.")
        return

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

    # 1. Daily trades
    daily_trades = get_trades_for_date(events, target_date)
    print(f"\n📅 Trades for {target_date.strftime('%A, %B %d, %Y')}")
    print("=" * 60)
    print_trade_table(daily_trades)
    print_summary(daily_trades, f"DAILY SUMMARY – {target_date.strftime('%Y-%m-%d')}")

    # 2. Month‑to‑date trades (from 1st of month to target_date)
    month_trades = get_trades_for_month(events, target_date)
    month_start = target_date.replace(day=1)
    print_summary(month_trades, f"MONTH‑TO‑DATE SUMMARY ({month_start.strftime('%b %d')} – {target_date.strftime('%b %d, %Y')})")

if __name__ == "__main__":
    main()