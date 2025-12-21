"""AutoCoinMarginBot 실행 엔트리포인트.

dev: Bybit 과거 캔들 기반 백테스트(텔레그램 요약, 파일/콘솔 로그).
prod: 실거래 루프(dry_run 기본, --live 시 주문 전송). logs/에 연도별 파일 기록.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from alerts import TelegramAlerter
from clients import BybitClient
from config import AppSettings, load_settings
from core import Backtester, Trader
from strategy.anchored_vwap import AnchoredVWAPStrategy


async def run_bot(settings: AppSettings, iterations: int, alerter: TelegramAlerter) -> None:
    """prod 모드에서 실거래 루프를 실행한다(기본 dry_run)."""
    _setup_logging(settings)
    iter_info = "inf" if iterations <= 0 else str(iterations)
    logger.info(
        f"설정 로드 완료 (dry_run={settings.dry_run}, env={settings.exchange.env}, iterations={iter_info})"
    )

    strategy = AnchoredVWAPStrategy(settings)
    client = BybitClient(settings)
    trader = Trader(settings, client=client, strategy=strategy, alerter=alerter)

    try:
        for symbol in settings.symbols:
            if not settings.dry_run:
                await client.set_leverage(symbol, settings.leverage)
                await client.set_margin_mode(symbol, settings.margin_mode)

        executed = 0
        while iterations <= 0 or executed < iterations:
            for symbol in settings.symbols:
                ticker = await client.get_ticker(symbol)
                await trader.handle_ticker(ticker)
            await asyncio.sleep(1)
            executed += 1
    finally:
        await client.close()


def _setup_logging(settings: AppSettings) -> None:
    """로깅 레벨과 출력 스트림을 설정한다."""
    logger.remove()
    logger.add(sys.stdout, level=settings.log_level, enqueue=True, backtrace=False)
    # 로그 디렉토리 생성 및 연도별 파일 로깅
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    year_suffix = datetime.now().strftime("%Y")
    log_path = log_dir / f"autocoinmarginbot_{year_suffix}.log"
    logger.add(log_path, level=settings.log_level, rotation="1 year", enqueue=True, backtrace=False)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="AutoCoinMarginBot runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (기본: ./config.yaml 존재 시 자동 사용)")
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Polling iterations (<=0 이면 무한 루프)",
    )
    parser.add_argument("--live", action="store_true", help="Disable dry-run and send live orders (prod only)")
    return parser.parse_args()


async def run_backtest(settings: AppSettings, alerter: TelegramAlerter) -> None:
    """dev 모드 백테스트 실행(REST kline, backtest_limit/start/end 반영, 텔레그램 요약 전송)."""
    _setup_logging(settings)
    logger.info("dev 모드 백테스트 실행 (드라이런)")
    strategy = AnchoredVWAPStrategy(settings)
    backtester = Backtester(settings, strategy, alerter=alerter)
    await backtester.run()


async def async_main() -> None:
    """엔트리 비동기 실행."""
    args = parse_args()
    # --config 미지정 시 현재 경로의 config.yaml이 존재하면 자동 사용
    cfg_path = args.config
    if cfg_path is None and Path("config.yaml").exists():
        cfg_path = "config.yaml"
    alerter: TelegramAlerter | None = None
    try:
        settings = load_settings(cfg_path)
        mode = (settings.mode or "dev").lower()
        if mode == "prod":
            if args.live:
                settings.dry_run = False  # type: ignore[pydantic-field]
            else:
                # prod 모드지만 --live 미사용 시 안전하게 드라이런 유지
                settings.dry_run = True  # type: ignore[pydantic-field]

        alerter = TelegramAlerter(settings.telegram)
        # 시작 알림
        await alerter.send(
            f"[시작] 모드={settings.mode} env={settings.exchange.env} dry_run={settings.dry_run} symbols={settings.symbols}"
        )

        if mode == "prod":
            await run_bot(settings, args.iterations, alerter)
        else:
            # dev 모드 -> 백테스트 실행
            await run_backtest(settings, alerter)
    except Exception as exc:  # pylint: disable=broad-except
        if alerter:
            try:
                await alerter.send(f"[오류] 봇 중단: {exc}")
            except Exception:
                pass
        raise
    finally:
        if alerter:
            try:
                await alerter.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(async_main())
