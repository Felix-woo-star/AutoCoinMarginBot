"""포지션 상태 머신과 주문 실행 로직.

영/한: 앵커드 VWAP 시그널 기반 진입/부분익절/손절/청산, 수익률 및 현금/코인 잔고 로깅.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
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
        self._trend_block_ts: Dict[str, float] = {}
        self._trend_warn_ts: Dict[str, float] = {}
        self._trend_ema_cache: Dict[str, Tuple[float, float, float]] = {}
        self._regime_warn_ts: Dict[str, float] = {}
        self._regime_state: Dict[str, str] = {}
        self._regime_log_ts: Dict[str, float] = {}
        self._signal_history: Dict[str, deque[Signal]] = {}

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

        ema_pair = await self._get_trend_ema(symbol)
        if ema_pair:
            ticker = dict(ticker)
            ticker["emaFast"] = ema_pair[0]
            ticker["emaSlow"] = ema_pair[1]
        vwap_signal = self.strategy.evaluate(ticker)
        signal = await self._select_regime_signal(symbol, price, vwap_signal)
        self._record_signal(symbol, signal)
        if pos.is_flat:
            if signal == Signal.LONG_ENTRY:
                if self._confirm_signal(symbol, signal) and await self._trend_allows(symbol, signal):
                    await self._open_position(symbol, "LONG", price)
            elif signal == Signal.SHORT_ENTRY:
                if self._confirm_signal(symbol, signal) and await self._trend_allows(symbol, signal):
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

    async def _trend_allows(self, symbol: str, signal: Signal) -> bool:
        """EMA 크로스 추세 필터로 진입을 제한한다."""
        if self.settings.strategy.regime.enabled:
            return True
        tf = self.settings.strategy.trend_filter
        if not tf.enabled:
            return True
        if signal not in (Signal.LONG_ENTRY, Signal.SHORT_ENTRY):
            return True
        ema_pair = await self._get_trend_ema(symbol)
        if not ema_pair:
            return False

        ema_fast, ema_slow = ema_pair
        allow = ema_fast > ema_slow if signal == Signal.LONG_ENTRY else ema_fast < ema_slow
        if not allow:
            now = time.time()
            last_ts = self._trend_block_ts.get(symbol, 0.0)
            if now - last_ts >= 300:
                self._trend_block_ts[symbol] = now
                logger.info(
                    "[추세필터] 진입 차단 symbol={} signal={} ema_fast={:.6f} ema_slow={:.6f}",
                    symbol,
                    signal.name,
                    ema_fast,
                    ema_slow,
                )
        return allow

    def _record_signal(self, symbol: str, signal: Signal) -> None:
        sc = self.settings.strategy.signal_confirm
        if not sc.enabled:
            return
        window = max(1, int(sc.window))
        history = self._signal_history.get(symbol)
        if not history or history.maxlen != window:
            history = deque(maxlen=window)
            self._signal_history[symbol] = history
        history.append(signal)

    def _confirm_signal(self, symbol: str, signal: Signal) -> bool:
        sc = self.settings.strategy.signal_confirm
        if not sc.enabled:
            return True
        if signal not in (Signal.LONG_ENTRY, Signal.SHORT_ENTRY):
            return False
        history = self._signal_history.get(symbol)
        if not history or len(history) < max(1, int(sc.window)):
            return False
        required = max(1, int(sc.required))
        required = min(required, len(history))
        target = Signal.LONG_ENTRY if signal == Signal.LONG_ENTRY else Signal.SHORT_ENTRY
        return sum(1 for s in history if s == target) >= required

    def _reset_signal_history(self, symbol: str) -> None:
        if symbol in self._signal_history:
            self._signal_history[symbol].clear()

    async def _get_trend_ema(self, symbol: str) -> Optional[Tuple[float, float]]:
        tf = self.settings.strategy.trend_filter
        if not tf.enabled:
            return None
        now = time.time()
        cached = self._trend_ema_cache.get(symbol)
        if cached and (now - cached[0] < tf.ema_refresh_sec):
            return (cached[1], cached[2])
        ema_pair = await self.client.get_ema_pair(
            symbol=symbol,
            interval=tf.ema_interval,
            fast=tf.ema_fast,
            slow=tf.ema_slow,
            lookback=tf.ema_lookback,
            refresh_sec=tf.ema_refresh_sec,
        )
        if ema_pair:
            self._trend_ema_cache[symbol] = (now, ema_pair[0], ema_pair[1])
            return ema_pair
        last_ts = self._trend_warn_ts.get(symbol, 0.0)
        if now - last_ts >= 300:
            self._trend_warn_ts[symbol] = now
            logger.warning("[추세필터] EMA 조회 실패로 진입을 보류합니다 symbol={}", symbol)
        return None

    async def _get_regime_metrics(self, symbol: str) -> Optional[Dict[str, float]]:
        reg = self.settings.strategy.regime
        if not reg.enabled:
            return None
        metrics = await self.client.get_regime_metrics(
            symbol=symbol,
            interval=reg.interval,
            ema_fast=reg.ema_fast,
            ema_slow=reg.ema_slow,
            atr_period=reg.atr_period,
            lookback=reg.lookback,
            refresh_sec=reg.refresh_sec,
        )
        if metrics:
            return metrics
        now = time.time()
        last_ts = self._regime_warn_ts.get(symbol, 0.0)
        if now - last_ts >= 300:
            self._regime_warn_ts[symbol] = now
            logger.warning("[레짐] 지표 조회 실패로 VWAP 전략만 사용합니다 symbol={}", symbol)
        return None

    def _classify_regime(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        atr: float,
    ) -> Tuple[str, float, float]:
        reg = self.settings.strategy.regime
        price_ref = price if price > 0 else 1.0
        ema_spread = abs(ema_fast - ema_slow) / price_ref
        atr_pct = atr / price_ref
        if ema_spread >= reg.ema_spread_trend_pct and atr_pct >= reg.atr_trend_pct:
            regime = "trend"
        elif ema_spread <= reg.ema_spread_range_pct and atr_pct <= reg.atr_range_pct:
            regime = "range"
        else:
            regime = "neutral"
        return regime, ema_spread, atr_pct

    async def _select_regime_signal(self, symbol: str, price: float, vwap_signal: Signal) -> Signal:
        reg = self.settings.strategy.regime
        if not reg.enabled:
            return vwap_signal
        metrics = await self._get_regime_metrics(symbol)
        if not metrics:
            return vwap_signal
        try:
            ema_fast = float(metrics["ema_fast"])
            ema_slow = float(metrics["ema_slow"])
            atr = float(metrics["atr"])
            price_ref = float(metrics.get("price") or price or 0.0)
        except (TypeError, ValueError, KeyError):
            return vwap_signal
        if price_ref <= 0:
            return vwap_signal

        regime, ema_spread, atr_pct = self._classify_regime(price_ref, ema_fast, ema_slow, atr)
        if ema_fast > ema_slow:
            trend_signal = Signal.LONG_ENTRY
        elif ema_fast < ema_slow:
            trend_signal = Signal.SHORT_ENTRY
        else:
            trend_signal = Signal.NONE

        if regime == "trend":
            signal = trend_signal
            strategy_label = "trend"
        elif regime == "range":
            signal = vwap_signal
            strategy_label = "vwap"
        else:
            behavior = (reg.neutral_behavior or "vwap").lower()
            if behavior == "trend":
                signal = trend_signal
                strategy_label = "trend"
            elif behavior == "none":
                signal = Signal.NONE
                strategy_label = "none"
            else:
                signal = vwap_signal
                strategy_label = "vwap"

        self._log_regime(symbol, regime, strategy_label, ema_fast, ema_slow, ema_spread, atr, atr_pct, price_ref)
        return signal

    def _log_regime(
        self,
        symbol: str,
        regime: str,
        strategy_label: str,
        ema_fast: float,
        ema_slow: float,
        ema_spread: float,
        atr: float,
        atr_pct: float,
        price: float,
    ) -> None:
        now = time.time()
        last_state = self._regime_state.get(symbol)
        last_ts = self._regime_log_ts.get(symbol, 0.0)
        if regime != last_state or now - last_ts >= 300:
            self._regime_state[symbol] = regime
            self._regime_log_ts[symbol] = now
            logger.info(
                "[레짐] symbol={} regime={} strategy={} ema_fast={:.6f} ema_slow={:.6f} spread={:.6f} atr={:.6f} atr_pct={:.6f} price={:.4f}",
                symbol,
                regime,
                strategy_label,
                ema_fast,
                ema_slow,
                ema_spread,
                atr,
                atr_pct,
                price,
            )

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
        self._reset_signal_history(symbol)
        logger.info(f"[잔고] 현금={self.cash_balance:.2f} USDT, 코인={self.coin_balance[symbol]:.6f} {symbol}")
        await self.alerter.send(f"*Enter {side}* {symbol} qty={qty} @ {price}")
        await self._log_wallet_balance(note="enter")

    async def _manage_long(self, pos: PositionState, price: float) -> None:
        """롱 포지션 TP/SL 관리(TP는 take_profit_move_long)."""
        tp1 = self.settings.strategy.take_profit_move_long
        tp_level = pos.entry_price * (1 + tp1)
        sl_level = pos.entry_price * (1 - self.settings.strategy.stop_loss_long)
        moved_stop = False
        if pos.partial_taken and self.settings.strategy.move_stop_to_entry_after_partial:
            breakeven = pos.entry_price * (1 + self.settings.strategy.break_even_buffer_pct)
            if breakeven > sl_level:
                sl_level = breakeven
                moved_stop = True
        if price <= sl_level:
            reason = "Long breakeven stop hit" if moved_stop else "Long stop-loss hit"
            await self._close_position(pos, price, reason=reason)
            return
        if not pos.partial_taken and price >= tp_level:
            tp_pct = tp1 * 100
            await self._partial_take_profit(pos, price, reason=f"Long TP1 +{tp_pct:.2f}%")
            return
        if pos.partial_taken:
            tp2 = self.settings.strategy.take_profit_move_long2
            if tp2 is None:
                tp2 = tp1 * 2
            if tp2 > 0:
                tp2_level = pos.entry_price * (1 + tp2)
                if price >= tp2_level:
                    tp2_pct = tp2 * 100
                    await self._close_position(pos, price, reason=f"Long TP2 +{tp2_pct:.2f}%")
                    return

    async def _manage_short(self, pos: PositionState, price: float) -> None:
        """숏 포지션 TP/SL 관리(TP는 take_profit_move_short)."""
        tp1 = self.settings.strategy.take_profit_move_short
        tp_level = pos.entry_price * (1 - tp1)
        sl_level = pos.entry_price * (1 + self.settings.strategy.stop_loss_short)
        moved_stop = False
        if pos.partial_taken and self.settings.strategy.move_stop_to_entry_after_partial:
            breakeven = pos.entry_price * (1 - self.settings.strategy.break_even_buffer_pct)
            if breakeven < sl_level:
                sl_level = breakeven
                moved_stop = True
        if price >= sl_level:
            reason = "Short breakeven stop hit" if moved_stop else "Short stop-loss hit"
            await self._close_position(pos, price, reason=reason)
            return
        if not pos.partial_taken and price <= tp_level:
            tp_pct = tp1 * 100
            await self._partial_take_profit(pos, price, reason=f"Short TP1 -{tp_pct:.2f}%")
            return
        if pos.partial_taken:
            tp2 = self.settings.strategy.take_profit_move_short2
            if tp2 is None:
                tp2 = tp1 * 2
            if tp2 > 0:
                tp2_level = pos.entry_price * (1 - tp2)
                if price <= tp2_level:
                    tp2_pct = tp2 * 100
                    await self._close_position(pos, price, reason=f"Short TP2 -{tp2_pct:.2f}%")
                    return

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
        self._reset_signal_history(pos.symbol)
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
        min_interval = max(1, int(self.settings.wallet_balance_log_interval_sec))
        if now - self._last_balance_ts < min_interval:
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
