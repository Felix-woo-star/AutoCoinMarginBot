#!/usr/bin/env python3
"""Grid search for backtest parameters."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from itertools import product
from typing import List, Tuple

from loguru import logger

from config import load_settings
from core.backtest import Backtester
from strategy.anchored_vwap import AnchoredVWAPStrategy


def _parse_float_list(raw: str) -> List[float]:
    raw = raw.strip()
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid range format: {raw}")
        start, end, step = (float(p) for p in parts)
        if step <= 0:
            raise ValueError("Step must be positive")
        values = []
        cur = start
        while cur <= end + 1e-12:
            values.append(round(cur, 6))
            cur += step
        return values
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    raw = raw.strip()
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid range format: {raw}")
        start, end, step = (int(p) for p in parts)
        if step <= 0:
            raise ValueError("Step must be positive")
        return list(range(start, end + 1, step))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_interval_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _summarize(backtester: Backtester, start_eq: float) -> dict:
    trades = backtester.trades
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0
    end_eq = backtester.equity_curve[-1] if backtester.equity_curve else start_eq
    total_return_pct = ((end_eq - start_eq) / start_eq) * 100.0 if start_eq else 0.0
    max_dd = Backtester._max_drawdown(backtester.equity_curve) if backtester.equity_curve else 0.0
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "return_pct": total_return_pct,
        "max_dd": max_dd,
        "final_cash": end_eq,
    }


async def _run_case(settings, k: float, interval: str, lookback: int) -> dict:
    settings.strategy.band.k = k
    settings.strategy.band.candle_interval = interval
    settings.strategy.band.vwap_lookback = lookback
    strategy = AnchoredVWAPStrategy(settings)
    backtester = Backtester(settings, strategy, alerter=None)
    await backtester.run()
    summary = _summarize(backtester, settings.initial_balance_usdt)
    summary.update({"k": k, "interval": interval, "lookback": lookback})
    return summary


async def _run_all(args: argparse.Namespace) -> Tuple[List[dict], int]:
    base_settings = load_settings(args.config)
    base_settings.mode = "dev"
    base_settings.dry_run = True
    base_settings.telegram.enabled = False
    base_settings.strategy.band.use_api_vwap = args.use_api_vwap
    if args.backtest_limit is not None:
        base_settings.backtest_limit = args.backtest_limit

    k_values = _parse_float_list(args.k) if args.k else [base_settings.strategy.band.k]
    intervals = _parse_interval_list(args.candle_interval) if args.candle_interval else [
        base_settings.strategy.band.candle_interval
    ]
    lookbacks = _parse_int_list(args.vwap_lookback) if args.vwap_lookback else [
        base_settings.strategy.band.vwap_lookback
    ]

    total = len(k_values) * len(intervals) * len(lookbacks)
    print(
        f"Grid size: k={len(k_values)} intervals={len(intervals)} lookbacks={len(lookbacks)} total={total}",
        flush=True,
    )
    results = []
    idx = 0
    for k, interval, lookback in product(k_values, intervals, lookbacks):
        idx += 1
        print(f"Running case {idx}/{total}...", flush=True)
        settings = base_settings.model_copy(deep=True)
        result = await _run_case(settings, k, interval, lookback)
        print(
            f"[{idx}/{total}] k={k} interval={interval} lookback={lookback} "
            f"trades={result['trades']} return={result['return_pct']:.2f}% win={result['win_rate']:.2f}%",
            flush=True,
        )
        results.append(result)
    return results, total


def _print_table(rows: List[dict], top: int) -> None:
    if not rows or top <= 0:
        print("\nNo results to display.", flush=True)
        return
    headers = ["k", "interval", "lookback", "trades", "win_rate", "return_pct", "max_dd", "final_cash"]
    formatted = []
    for r in rows[:top]:
        formatted.append(
            [
                f"{r['k']:.6f}",
                r["interval"],
                str(r["lookback"]),
                str(r["trades"]),
                f"{r['win_rate']:.2f}",
                f"{r['return_pct']:.2f}",
                f"{r['max_dd']:.2f}",
                f"{r['final_cash']:.2f}",
            ]
        )
    widths = [len(h) for h in headers]
    for row in formatted:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    print("\nTop results", flush=True)
    print(header_line, flush=True)
    print(sep_line, flush=True)
    for row in formatted:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)), flush=True)


def _write_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["k", "interval", "lookback", "trades", "win_rate", "return_pct", "max_dd", "final_cash"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest grid search")
    parser.add_argument("--config", type=str, default="config.backtest.yaml", help="Base config path")
    parser.add_argument("--k", type=str, default=None, help="Comma list or range: start:end:step")
    parser.add_argument("--candle-interval", type=str, default=None, help="Comma list (e.g., 5m,15m,30m)")
    parser.add_argument("--vwap-lookback", type=str, default=None, help="Comma list or range: start:end:step")
    parser.add_argument("--backtest-limit", type=int, default=None, help="Override backtest_limit")
    parser.add_argument("--use-api-vwap", action="store_true", help="Use HLC avg as API vwap proxy")
    parser.add_argument("--top", type=int, default=10, help="How many top rows to display")
    parser.add_argument("--out", type=str, default=None, help="Write full results to CSV")
    parser.add_argument("--log-level", type=str, default="WARNING", help="Logger level for backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    results, total = asyncio.run(_run_all(args))
    results_sorted = sorted(
        results,
        key=lambda r: (r["return_pct"], -r["max_dd"]),
        reverse=True,
    )
    _print_table(results_sorted, min(args.top, total))
    if args.out:
        _write_csv(results_sorted, args.out)
        print(f"\nSaved CSV: {args.out}", flush=True)


if __name__ == "__main__":
    main()
