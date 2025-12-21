"""백테스트 실행기: Bybit 히스토리컬 캔들을 재생하며 손익/잔고/수익률을 계산한다.

- 입력: backtest_limit 또는 start/end, kline interval, 전략 파라미터.
- 출력: 거래별 PnL, 총 수익률, MDD, 현금/코인 잔고, 텔레그램 요약(옵션).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from loguru import logger

from alerts import TelegramAlerter
from config import AppSettings
from strategy.anchored_vwap import AnchoredVWAPStrategy, Signal


@dataclass
class Trade:
    """체결 기록."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    partial: bool


class Backtester:
    """경량 백테스트: 시그널 재생 + 단순 체결/손익/리포트."""

    def __init__(
        self,
        settings: AppSettings,
        strategy: AnchoredVWAPStrategy,
        alerter: Optional[TelegramAlerter] = None,
    ) -> None:
        self.settings = settings
        self.strategy = strategy
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.cash_balance: float = settings.initial_balance_usdt
        self.coin_balance: Dict[str, float] = {sym: 0.0 for sym in settings.symbols}
        self.alerter = alerter

    async def run(self) -> None:
        """히스토리컬 캔들을 가져와 시뮬레이션한다."""
        symbol = self.settings.symbols[0]
        candles = await self._fetch_klines(symbol)
        if not candles:
            logger.error("캔들 데이터를 가져오지 못했습니다. 인터넷 연결을 확인하거나 interval/심볼을 점검하세요.")
            return
        await self._simulate(symbol, candles)
        self._report()
        await self._notify_summary()

    async def _fetch_klines(self, symbol: str, limit: Optional[int] = None) -> List[dict]:
        """Bybit 공개 REST로 히스토리컬 kline 조회."""
        interval_raw = self.settings.strategy.band.candle_interval
        interval = self._normalize_interval(interval_raw)
        limit_val = limit or self.settings.backtest_limit
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit_val}
        if self.settings.backtest_start:
            start_ms = self._parse_time_ms(self.settings.backtest_start)
            if start_ms:
                params["start"] = start_ms
        if self.settings.backtest_end:
            end_ms = self._parse_time_ms(self.settings.backtest_end)
            if end_ms:
                params["end"] = end_ms
        logger.info(
            f"[BT] 캔들 조회: 심볼={symbol} interval={interval_raw}->{interval} limit={params['limit']} start={params.get('start')} end={params.get('end')}"
        )
        url = f"{self.settings.exchange.rest_base}/v5/market/kline"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, params=params)
                res.raise_for_status()
                data = res.json()
        except Exception as exc:  # pragma: no cover - 네트워크 예외
            logger.error(f"kline 조회 실패: {exc}")
            return []

        if data.get("retCode") != 0:
            logger.error(f"Bybit 응답 오류: {data}")
            return []

        klines = data.get("result", {}).get("list") or []
        # Bybit v5 kline 포맷: [start, open, high, low, close, volume, turnover]
        parsed = [
            {
                "start": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "turnover": float(k[6]),
            }
            for k in klines
        ]
        parsed.sort(key=lambda x: x["start"])
        if parsed:
            start_iso = self._fmt_time(parsed[0]["start"])
            end_iso = self._fmt_time(parsed[-1]["start"])
            logger.info(f"[BT] 캔들 구간: {start_iso} ~ {end_iso} (개수={len(parsed)})")
        return parsed

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        """Bybit kline interval 포맷으로 변환(예: '1h' -> '60', '1m' -> '1', 'D' -> 'D')."""
        s = str(interval).strip()
        sl = s.lower()
        if sl.endswith("h"):
            num = int(sl.replace("h", ""))
            return str(num * 60)
        if sl.endswith("m"):
            return sl.replace("m", "")
        if sl in ("d", "w", "m"):
            return s.upper()
        if s.endswith("h"):
            num = int(s.replace("h", ""))
            return str(num * 60)
        if s.endswith("m"):
            return s.replace("m", "")
        return s

    async def _simulate(self, symbol: str, candles: List[dict]) -> None:
        """캔들을 순회하며 시그널/포지션/손익을 계산(TP는 롱/숏 분리 설정 반영)."""
        side: Optional[str] = None
        entry_price: Optional[float] = None
        qty: float = 0.0
        partial_taken = False
        equity = self.settings.initial_balance_usdt
        fee_rate = 0.0006  # 단순 테이커 수수료 가정
        lookback = max(1, self.settings.strategy.band.vwap_lookback)
        use_api_vwap = self.settings.strategy.band.use_api_vwap
        vwap_window = deque()
        sum_volume = 0.0
        sum_turnover = 0.0

        for c in candles:
            close = c["close"]
            volume = float(c.get("volume") or 0.0)
            turnover = float(c.get("turnover") or 0.0)
            if volume > 0 and turnover > 0:
                vwap_window.append((volume, turnover))
                sum_volume += volume
                sum_turnover += turnover
                if len(vwap_window) > lookback:
                    old_volume, old_turnover = vwap_window.popleft()
                    sum_volume -= old_volume
                    sum_turnover -= old_turnover
            local_vwap = (sum_turnover / sum_volume) if sum_volume > 0 else 0.0

            ticker = {"symbol": symbol, "lastPrice": close, "markPrice": close}
            if use_api_vwap:
                # 백테스트에는 API avgPrice가 없으므로 HLC 평균으로 대체
                avg_price = (c["high"] + c["low"] + close) / 3
                ticker["avgPrice"] = avg_price
            else:
                if local_vwap:
                    ticker["vwap"] = local_vwap
            signal = self.strategy.evaluate(ticker)
            ts_str = self._fmt_time(c["start"])

            # 포지션이 없을 때: 시그널 진입
            if side is None:
                if signal == Signal.LONG_ENTRY:
                    side = "LONG"
                    entry_price = close
                    qty = self._calc_qty(close)
                    partial_taken = False
                    self.coin_balance[symbol] = qty
                    logger.info(
                        f"[BT_진입] {side} {symbol} 수량={qty:.6f} 가격={close:.4f} 시각={ts_str} | 현금={self.cash_balance:.2f}USDT 코인={self.coin_balance[symbol]:.6f}"
                    )
                elif signal == Signal.SHORT_ENTRY:
                    side = "SHORT"
                    entry_price = close
                    qty = self._calc_qty(close)
                    partial_taken = False
                    self.coin_balance[symbol] = -qty
                    logger.info(
                        f"[BT_진입] {side} {symbol} 수량={qty:.6f} 가격={close:.4f} 시각={ts_str} | 현금={self.cash_balance:.2f}USDT 코인={self.coin_balance[symbol]:.6f}"
                    )
                self.equity_curve.append(equity)
                continue

            # 포지션 보유 시 TP/SL 우선, 반대 시그널은 종가 청산
            if side == "LONG":
                tp = entry_price * (1 + self.settings.strategy.take_profit_move_long)
                sl = entry_price * (1 - self.settings.strategy.stop_loss_long)
                exit_price = None
                # 보수적으로 SL 우선 처리
                if c["low"] <= sl:
                    exit_price = sl
                elif (not partial_taken) and c["high"] >= tp:
                    # 부분 익절 50%, 남은 물량 유지
                    realized = (tp - entry_price) * (qty * self.settings.strategy.partial_take_profit_pct)
                    fee = tp * qty * self.settings.strategy.partial_take_profit_pct * fee_rate * 2
                    equity += realized - fee
                    self.cash_balance += realized - fee
                    partial_qty = qty * self.settings.strategy.partial_take_profit_pct
                    remain_qty = qty * (1 - self.settings.strategy.partial_take_profit_pct)
                    self.coin_balance[symbol] = remain_qty
                    logger.info(
                        f"[BT_부분익절] {side} {symbol} 익절수량={partial_qty:.6f} 가격={tp:.4f} 잔여수량={remain_qty:.6f} 시각={ts_str} | 현금={self.cash_balance:.2f}USDT 코인={self.coin_balance[symbol]:.6f}"
                    )
                    qty *= (1 - self.settings.strategy.partial_take_profit_pct)
                    partial_taken = True
                # 반대 시그널 시 종가 청산
                if signal == Signal.SHORT_ENTRY:
                    exit_price = close
                if exit_price and entry_price is not None:
                    pnl = (exit_price - entry_price) * qty
                    fee = exit_price * qty * fee_rate * 2
                    pnl -= fee
                    equity += pnl
                    self.cash_balance += pnl
                    self.coin_balance[symbol] = 0.0
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0.0
                    logger.info(
                        f"[BT_청산] LONG {symbol} 수량={qty:.6f} 진입={entry_price:.4f} 청산={exit_price:.4f} PnL={pnl:.4f} 수익률={pnl_pct:.2f}% 시각={ts_str} | 현금={self.cash_balance:.2f}USDT 코인={self.coin_balance[symbol]:.6f}"
                    )
                    self.trades.append(
                        Trade(
                            symbol=symbol,
                            side="LONG",
                            entry_price=entry_price,
                            exit_price=exit_price,
                            qty=qty,
                            pnl=pnl,
                            partial=partial_taken,
                        )
                    )
                    side, entry_price, qty, partial_taken = None, None, 0.0, False

            elif side == "SHORT":
                tp = entry_price * (1 - self.settings.strategy.take_profit_move_short)
                sl = entry_price * (1 + self.settings.strategy.stop_loss_short)
                exit_price = None
                if c["high"] >= sl:
                    exit_price = sl
                elif (not partial_taken) and c["low"] <= tp:
                    realized = (entry_price - tp) * (qty * self.settings.strategy.partial_take_profit_pct)
                    fee = tp * qty * self.settings.strategy.partial_take_profit_pct * fee_rate * 2
                    equity += realized - fee
                    self.cash_balance += realized - fee
                    partial_qty = qty * self.settings.strategy.partial_take_profit_pct
                    remain_qty = qty * (1 - self.settings.strategy.partial_take_profit_pct)
                    self.coin_balance[symbol] = -remain_qty
                    logger.info(
                        f"[BT_부분익절] {side} {symbol} 익절수량={partial_qty:.6f} 가격={tp:.4f} 잔여수량={remain_qty:.6f} 시각={ts_str} | 현금={self.cash_balance:.2f}USDT 코인={self.coin_balance[symbol]:.6f}"
                    )
                    qty *= (1 - self.settings.strategy.partial_take_profit_pct)
                    partial_taken = True
                if signal == Signal.LONG_ENTRY:
                    exit_price = close
                if exit_price and entry_price is not None:
                    pnl = (entry_price - exit_price) * qty
                    fee = exit_price * qty * fee_rate * 2
                    pnl -= fee
                    equity += pnl
                    self.cash_balance += pnl
                    self.coin_balance[symbol] = 0.0
                    pnl_pct = (entry_price - exit_price) / entry_price * 100 if entry_price else 0.0
                    logger.info(
                        f"[BT_청산] SHORT {symbol} 수량={qty:.6f} 진입={entry_price:.4f} 청산={exit_price:.4f} PnL={pnl:.4f} 수익률={pnl_pct:.2f}% 시각={ts_str} | 현금={self.cash_balance:.2f}USDT 코인={self.coin_balance[symbol]:.6f}"
                    )
                    self.trades.append(
                        Trade(
                            symbol=symbol,
                            side="SHORT",
                            entry_price=entry_price,
                            exit_price=exit_price,
                            qty=qty,
                            pnl=pnl,
                            partial=partial_taken,
                        )
                    )
                    side, entry_price, qty, partial_taken = None, None, 0.0, False

            self.equity_curve.append(equity)

    def _calc_qty(self, price: float) -> float:
        """주문 수량(USDT 기준)을 계산."""
        return round(self.settings.order_size_usdt / price, 6)

    def _report(self) -> None:
        """간단 성과 리포트 출력."""
        if not self.trades:
            logger.warning("백테스트 결과: 체결된 거래가 없습니다.")
            return
        total_pnl = sum(t.pnl for t in self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        win_rate = (len(wins) / len(self.trades)) * 100
        max_dd = self._max_drawdown(self.equity_curve)
        start_eq = self.settings.initial_balance_usdt
        end_eq = self.equity_curve[-1] if self.equity_curve else start_eq
        total_return_pct = ((end_eq - start_eq) / start_eq) * 100 if start_eq else 0.0
        logger.info(
            f"백테스트 완료 | 거래수={len(self.trades)} | 승률={win_rate:.2f}% | 총 PnL={total_pnl:.2f} | 총 수익률={total_return_pct:.2f}% | 최종 현금잔고={self.cash_balance:.2f} | MDD={max_dd:.2f}"
        )
        # 상위 몇 건 샘플 로그
        for t in self.trades[:5]:
            logger.debug(
                f"체결 {t.side} {t.symbol} 진입={t.entry_price:.4f} 청산={t.exit_price:.4f} 수량={t.qty:.4f} PnL={t.pnl:.4f} 부분익절여부={t.partial}"
            )

        # 결과를 메모리에 유지해 알림에 사용
        self._summary = {
            "trades": len(self.trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "max_dd": max_dd,
            "final_cash": self.cash_balance,
        }

    async def _notify_summary(self) -> None:
        """텔레그램으로 요약 전송(선택)."""
        if not self.alerter or not self.trades:
            return
        msg = (
            f"*백테스트 결과*\n"
            f"거래수: {self._summary['trades']}\n"
            f"승률: {self._summary['win_rate']:.2f}%\n"
            f"총 PnL: {self._summary['total_pnl']:.2f} USDT\n"
            f"총 수익률: {self._summary['total_return_pct']:.2f}%\n"
            f"MDD: {self._summary['max_dd']:.2f}\n"
            f"최종 현금잔고: {self._summary['final_cash']:.2f} USDT"
        )
        await self.alerter.send(msg)

    @staticmethod
    def _max_drawdown(curve: List[float]) -> float:
        """최대 낙폭을 계산한다."""
        peak = float("-inf")
        max_dd = 0.0
        for eq in curve:
            peak = max(peak, eq)
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _fmt_time(ts_ms: int) -> str:
        """밀리초 기준 타임스탬프를 ISO 문자열로 변환."""
        return datetime.fromtimestamp(ts_ms / 1000).isoformat()

    @staticmethod
    def _parse_time_ms(value: str) -> Optional[int]:
        """ISO 문자열을 ms 타임스탬프로 변환."""
        try:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        except Exception:
            logger.warning(f"[BT] 시간 파싱 실패: {value}")
            return None
