"""거래소 VWAP을 사용하는 앵커드 VWAP 전략 시그널러.

가격- VWAP 프리미엄/디스카운트를 k 밴드로 해석해 롱/숏 진입 시그널을 만든다.
롱 +3%/숏 -3% 부분익절, 롱 -1%/숏 +1% 손절 규칙과 함께 사용된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import Any, Dict, Optional

from config import AppSettings
from loguru import logger


class Signal(Enum):
    NONE = auto()
    LONG_ENTRY = auto()
    SHORT_ENTRY = auto()
    LONG_EXIT = auto()
    SHORT_EXIT = auto()


@dataclass
class StrategyContext:
    anchor_vwap: Optional[float] = None
    anchor_label: str = "initial"
    last_debug_ts: float = 0.0


class AnchoredVWAPStrategy:
    """
    거래소 VWAP 값을 그대로 사용하여 프리미엄/디스카운트로 방향성을 판단한다.
    로컬 VWAP 계산은 수행하지 않는다. 롱/숏 TP 임계치는 분리 설정.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.ctx = StrategyContext()

    def reset_anchor(self, anchor_vwap: float, label: str) -> None:
        self.ctx.anchor_vwap = anchor_vwap
        self.ctx.anchor_label = label

    def evaluate(self, ticker: Dict[str, Any]) -> Signal:
        """VWAP/avgPrice가 포함된 티커로 시그널을 계산한다."""
        symbol = ticker.get("symbol") or "UNKNOWN"
        last_price = ticker.get("lastPrice")
        mark_price = ticker.get("markPrice")
        avg_price = ticker.get("avgPrice")
        vwap_price = ticker.get("vwap")
        price = float(last_price or mark_price or 0.0)
        vwap = float(avg_price or vwap_price or 0.0)

        now = time.time()
        if now - self.ctx.last_debug_ts >= 60:
            if price and vwap:
                premium = (price - vwap) / vwap
                logger.info(
                    "[DEBUG] ticker symbol={} lastPrice={} markPrice={} avgPrice={} vwap={} premium={:.6f}",
                    symbol,
                    last_price,
                    mark_price,
                    avg_price,
                    vwap_price,
                    premium,
                )
            else:
                logger.info(
                    "[DEBUG] ticker missing price/vwap symbol={} lastPrice={} markPrice={} avgPrice={} vwap={}",
                    symbol,
                    last_price,
                    mark_price,
                    avg_price,
                    vwap_price,
                )
            self.ctx.last_debug_ts = now

        if not price or not vwap:
            return Signal.NONE

        # 앵커 미설정 시 초기 VWAP을 앵커로 설정
        if self.ctx.anchor_vwap is None:
            self.reset_anchor(anchor_vwap=vwap, label="initial")

        premium = (price - vwap) / vwap
        band = self.settings.strategy.band

        # 기본 방향성: VWAP 이상은 강세, 이하는 약세로 해석.
        # 밴드 폭 k는 노이즈를 줄이기 위한 최소 임계값 역할.
        threshold = band.k if band.k < 1 else band.k / 100.0

        if premium >= threshold:
            return Signal.LONG_ENTRY
        if premium <= -threshold:
            return Signal.SHORT_ENTRY
        return Signal.NONE
