"""포지션 상태 머신과 주문 실행 로직.

영/한: 앵커드 VWAP 시그널 기반 진입/부분익절/손절/청산, 수익률 및 현금/코인 잔고 로깅.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import time
from typing import Dict, Optional, Tuple

from loguru import logger

from alerts import TelegramAlerter
from clients import BybitClient
from config import AppSettings
from strategy.anchored_vwap import AnchoredVWAPStrategy, Signal


@dataclass
class PositionState:
    symbol: str
    side: Optional[str] = None  # "LONG" | "SHORT" (롱/숏)
    entry_price: Optional[float] = None
    size: float = 0.0
    partial_taken: bool = False

    @property
    def is_flat(self) -> bool:
        return self.side is None


class Trader:
    """PRD상의 진입/익절/손절/청산 규칙을 수행하는 상태 머신."""

    def __init__(
        self,
        settings: AppSettings,
        client: BybitClient,
        strategy: AnchoredVWAPStrategy,
        alerter: TelegramAlerter,
    ) -> None:
        self.settings = settings
        self.client = client
        self.strategy = strategy
        self.alerter = alerter
        self.positions: Dict[str, PositionState] = {sym: PositionState(symbol=sym) for sym in settings.symbols}
        self.cash_balance: float = settings.initial_balance_usdt
        self.coin_balance: Dict[str, float] = {sym: 0.0 for sym in settings.symbols}
        self._qty_filters: Dict[str, Tuple[Decimal, Decimal]] = {}
        self._min_qty_warn_ts: Dict[str, float] = {}
        self._last_balance_ts: float = 0.0

    async def handle_ticker(self, ticker: dict) -> None:
        """티커 1건을 받아 포지션 상태를 갱신한다."""
        symbol = ticker.get("symbol")
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        price = float(ticker.get("lastPrice") or ticker.get("markPrice") or 0.0)
        if not price:
            return

        use_api_vwap = self.settings.strategy.band.use_api_vwap
        api_vwap = ticker.get("avgPrice") or ticker.get("vwap")
        has_api_vwap = False
        if api_vwap is not None:
            try:
                has_api_vwap = float(api_vwap) > 0
            except (TypeError, ValueError):
                has_api_vwap = False

        if not use_api_vwap or not has_api_vwap:
            local_vwap = await self.client.get_local_vwap(
                symbol=symbol,
                interval=self.settings.strategy.band.candle_interval,
                limit=self.settings.strategy.band.vwap_lookback,
                refresh_sec=self.settings.strategy.band.vwap_refresh_sec,
            )
            if local_vwap:
                ticker = dict(ticker)
                ticker["vwap"] = local_vwap

        signal = self.strategy.evaluate(ticker)
        if pos.is_flat:
            if signal == Signal.LONG_ENTRY:
                await self._open_position(symbol, "LONG", price)
            elif signal == Signal.SHORT_ENTRY:
                await self._open_position(symbol, "SHORT", price)
            return

        if pos.side == "LONG":
            if signal == Signal.SHORT_ENTRY:
                await self._close_position(pos, price, reason="Opposite signal after long")
                return
            await self._manage_long(pos, price)
        elif pos.side == "SHORT":
            if signal == Signal.LONG_ENTRY:
                await self._close_position(pos, price, reason="Opposite signal after short")
                return
            await self._manage_short(pos, price)

    async def sync_positions(self) -> None:
        """거래소 포지션을 읽어 내부 상태를 동기화한다."""
        for symbol in self.settings.symbols:
            positions = await self.client.get_positions(symbol=symbol)
            active = []
            for p in positions:
                try:
                    size = float(p.get("size") or 0)
                except (TypeError, ValueError):
                    size = 0.0
                if size > 0:
                    active.append((p, size))

            if not active:
                continue
            if len(active) > 1:
                msg = f"[포지션 동기화 실패] {symbol}에 여러 포지션이 있어 동기화를 중단합니다."
                logger.error(msg)
                await self.alerter.send(msg)
                raise RuntimeError(msg)

            p, size = active[0]
            side_raw = (p.get("side") or "").lower()
            side = "LONG" if side_raw in ("buy", "long") else "SHORT"
            try:
                entry_price = float(p.get("avgPrice") or p.get("entryPrice") or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            expected_qty = 0.0
            if entry_price > 0:
                expected_qty = round(self.settings.order_size_usdt / entry_price, 6)
            partial_taken = False
            if expected_qty > 0:
                threshold = expected_qty * (1 - self.settings.strategy.partial_take_profit_pct) + 1e-9
                partial_taken = size <= threshold

            pos = self.positions[symbol]
            pos.side = side
            pos.entry_price = entry_price or None
            pos.size = size
            pos.partial_taken = partial_taken
            self.coin_balance[symbol] = size if side == "LONG" else -size
            logger.info(
                "[포지션 동기화] {} side={} size={} entry_price={} partial_taken={}",
                symbol,
                side,
                size,
                entry_price,
                partial_taken,
            )

        await self._log_wallet_balance(note="sync")

    async def _open_position(self, symbol: str, side: str, price: float) -> None:
        """시장가로 신규 진입."""
        qty = self._calc_qty(price)
        qty = await self._apply_qty_filters(symbol, qty, price)
        if qty <= 0:
            return
        logger.info(f"[진입] {side} {symbol} 수량={qty:.6f} 가격={price:.4f} (dry_run={self.settings.dry_run})")
        if not self.settings.dry_run:
            await self.client.place_market_order(symbol=symbol, side="Buy" if side == "LONG" else "Sell", qty=qty)
        pos = self.positions[symbol]
        pos.side = side
        pos.entry_price = price
        pos.size = qty
        pos.partial_taken = False
        self.coin_balance[symbol] = qty if side == "LONG" else -qty
        logger.info(f"[잔고] 현금={self.cash_balance:.2f} USDT, 코인={self.coin_balance[symbol]:.6f} {symbol}")
        await self.alerter.send(f"*Enter {side}* {symbol} qty={qty} @ {price}")
        await self._log_wallet_balance(note="enter")

    async def _manage_long(self, pos: PositionState, price: float) -> None:
        """롱 포지션 TP/SL 관리(TP는 take_profit_move_long)."""
        tp_level = pos.entry_price * (1 + self.settings.strategy.take_profit_move_long)
        sl_level = pos.entry_price * (1 - self.settings.strategy.stop_loss_long)
        if price <= sl_level:
            await self._close_position(pos, price, reason="Long stop-loss hit")
            return
        if not pos.partial_taken and price >= tp_level:
            tp_pct = self.settings.strategy.take_profit_move_long * 100
            await self._partial_take_profit(pos, price, reason=f"Long TP1 +{tp_pct:.2f}%")

    async def _manage_short(self, pos: PositionState, price: float) -> None:
        """숏 포지션 TP/SL 관리(TP는 take_profit_move_short)."""
        tp_level = pos.entry_price * (1 - self.settings.strategy.take_profit_move_short)
        sl_level = pos.entry_price * (1 + self.settings.strategy.stop_loss_short)
        if price >= sl_level:
            await self._close_position(pos, price, reason="Short stop-loss hit")
            return
        if not pos.partial_taken and price <= tp_level:
            tp_pct = self.settings.strategy.take_profit_move_short * 100
            await self._partial_take_profit(pos, price, reason=f"Short TP1 -{tp_pct:.2f}%")

    async def _partial_take_profit(self, pos: PositionState, price: float, reason: str) -> None:
        """부분 익절 50% 실행."""
        if pos.size <= 0:
            return
        qty = pos.size * self.settings.strategy.partial_take_profit_pct
        if pos.entry_price:
            pnl_pct = (
                (price - pos.entry_price) / pos.entry_price * 100
                if pos.side == "LONG"
                else (pos.entry_price - price) / pos.entry_price * 100
            )
        else:
            pnl_pct = 0.0
        # 부분 익절 분량의 실현 손익을 현금에 반영
        pnl_realized = (
            (price - pos.entry_price) * qty if pos.side == "LONG" else (pos.entry_price - price) * qty
        )
        self.cash_balance += pnl_realized
        self.coin_balance[pos.symbol] += -qty if pos.side == "LONG" else qty
        logger.info(
            f"[부분익절] {pos.side} {pos.symbol} 수량={qty:.6f} 가격={price:.4f} 수익률={pnl_pct:.2f}% 사유={reason}"
        )
        if not self.settings.dry_run:
            await self.client.place_market_order(
                symbol=pos.symbol, side="Sell" if pos.side == "LONG" else "Buy", qty=qty, reduce_only=True
            )
        pos.size -= qty
        pos.partial_taken = True
        logger.info(f"[잔고] 현금={self.cash_balance:.2f} USDT, 코인={self.coin_balance[pos.symbol]:.6f} {pos.symbol}")
        await self.alerter.send(f"*Partial TP* {pos.side} {pos.symbol} qty={qty} @ {price} | {reason}")
        await self._log_wallet_balance(note="partial")

    async def _close_position(self, pos: PositionState, price: float, reason: str) -> None:
        """잔여 물량 전량 청산."""
        qty = pos.size
        if pos.entry_price:
            pnl_pct = (
                (price - pos.entry_price) / pos.entry_price * 100
                if pos.side == "LONG"
                else (pos.entry_price - price) / pos.entry_price * 100
            )
        else:
            pnl_pct = 0.0
        pnl_realized = (
            (price - pos.entry_price) * qty if pos.side == "LONG" else (pos.entry_price - price) * qty
        )
        self.cash_balance += pnl_realized
        logger.info(
            f"[청산] {pos.side} {pos.symbol} 수량={qty:.6f} 가격={price:.4f} 수익률={pnl_pct:.2f}% 사유={reason}"
        )
        if qty > 0 and not self.settings.dry_run:
            await self.client.place_market_order(
                symbol=pos.symbol, side="Sell" if pos.side == "LONG" else "Buy", qty=qty, reduce_only=True
            )
        await self.alerter.send(f"*Exit* {pos.side} {pos.symbol} qty={qty} @ {price} | {reason}")
        pos.side = None
        pos.entry_price = None
        pos.size = 0.0
        pos.partial_taken = False
        self.coin_balance[pos.symbol] = 0.0
        logger.info(f"[잔고] 현금={self.cash_balance:.2f} USDT, 코인={self.coin_balance[pos.symbol]:.6f} {pos.symbol}")
        await self._log_wallet_balance(note="exit")

    def _calc_qty(self, price: float) -> float:
        """주문 수량(계좌 USDT 기준)을 계산한다."""
        return round(self.settings.order_size_usdt / price, 6)

    async def _apply_qty_filters(self, symbol: str, qty: float, price: float) -> float:
        """거래소 최소 수량/스텝에 맞게 수량을 보정한다."""
        filters = await self._get_qty_filters(symbol)
        if not filters:
            return qty
        min_qty, step = filters
        qty_dec = Decimal(str(qty))
        if step > 0:
            qty_dec = (qty_dec // step) * step
        if qty_dec < min_qty:
            await self._warn_min_qty(symbol, qty, min_qty, step, price)
            return 0.0
        return float(qty_dec)

    async def _get_qty_filters(self, symbol: str) -> Optional[Tuple[Decimal, Decimal]]:
        cached = self._qty_filters.get(symbol)
        if cached:
            return cached
        lot = await self.client.get_lot_size_filter(symbol)
        if not lot:
            return None
        min_qty_str, step_str = lot
        try:
            min_qty = Decimal(min_qty_str)
            step = Decimal(step_str)
        except Exception:
            return None
        if min_qty <= 0 or step <= 0:
            return None
        self._qty_filters[symbol] = (min_qty, step)
        return self._qty_filters[symbol]

    async def _warn_min_qty(
        self, symbol: str, qty: float, min_qty: Decimal, step: Decimal, price: float
    ) -> None:
        now = time.time()
        last_ts = self._min_qty_warn_ts.get(symbol, 0.0)
        if now - last_ts < 1800:
            return
        self._min_qty_warn_ts[symbol] = now
        msg = (
            f"[주문 스킵] {symbol} 수량 {qty:.6f} < 최소수량 {min_qty} (step={step}). "
            f"price={price:.2f}, order_size_usdt를 늘려주세요."
        )
        logger.warning(msg)
        await self.alerter.send(msg)

    async def _log_wallet_balance(self, note: str) -> None:
        now = time.time()
        if now - self._last_balance_ts < 60:
            return
        self._last_balance_ts = now
        try:
            data = await self.client.get_wallet_balance()
        except Exception as exc:
            logger.warning("[실잔고] 조회 실패 note={} error={}", note, exc)
            return
        if not data:
            return
        total_available = data.get("totalAvailableBalance")
        total_wallet = data.get("totalWalletBalance")
        total_margin = data.get("totalMarginBalance")
        usdt_available = None
        usdt_wallet = None
        for coin in data.get("coin") or []:
            if coin.get("coin") == "USDT":
                usdt_available = coin.get("availableToWithdraw") or coin.get("availableToBorrow")
                usdt_wallet = coin.get("walletBalance") or coin.get("equity")
                break
        logger.info(
            "[실잔고] note={} totalAvailable={} totalWallet={} totalMargin={} usdtAvailable={} usdtWallet={}",
            note,
            total_available,
            total_wallet,
            total_margin,
            usdt_available,
            usdt_wallet,
        )
