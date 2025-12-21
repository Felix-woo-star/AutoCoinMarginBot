"""설정 로드 및 검증 모델 정의 (YAML 단일 소스 사용).

주요 옵션: dev/prod, 테스트넷/메인넷, 전략 파라미터, 백테스트 limit/start/end,
초기 잔고/주문 크기, 텔레그램 토큰/채팅 ID, 로그 레벨.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class AnchorSettings(BaseModel):
    """VWAP 기준(앵커) 설정 옵션."""

    anchor_type: str = Field(
        default="daily_open", description="daily_open | manual | swing_high | swing_low | event"
    )
    manual_time_iso: Optional[str] = Field(
        default=None, description="Manual anchor time in ISO format if anchor_type=manual"
    )
    manual_price: Optional[float] = Field(
        default=None, description="Manual anchor price if anchor_type=manual"
    )


class BandSettings(BaseModel):
    """VWAP 밴드 계산 파라미터."""

    method: str = Field(default="std", description="std | atr")
    k: float = Field(default=1.0, description="Band width multiplier")
    candle_interval: str = Field(default="1h", description="Interval for price/vol signals")
    use_api_vwap: bool = Field(default=True, description="Use exchange-provided VWAP instead of local calc")
    vwap_lookback: int = Field(default=200, description="Candles used for local VWAP")
    vwap_refresh_sec: int = Field(default=30, description="VWAP refresh interval in seconds")


class StrategySettings(BaseModel):
    anchor: AnchorSettings = Field(default_factory=AnchorSettings)
    band: BandSettings = Field(default_factory=BandSettings)
    partial_take_profit_pct: float = Field(default=0.5, description="Portion to take at first TP")
    take_profit_move_long: float = Field(default=0.03, description="First TP threshold for longs (fraction, 0.03=3%)")
    take_profit_move_short: float = Field(default=0.03, description="First TP threshold for shorts (fraction, 0.03=3%)")
    stop_loss_long: float = Field(default=0.01, description="Hard stop for longs (fraction, 0.01=1%)")
    stop_loss_short: float = Field(default=0.01, description="Hard stop for shorts (fraction, 0.01=1%)")


class ExchangeSettings(BaseModel):
    env: str = Field(default="testnet", description="testnet | mainnet")
    api_key: str = Field(default="", description="Bybit API key")
    api_secret: str = Field(default="", description="Bybit API secret")
    rest_base: str = Field(default="https://api-testnet.bybit.com")
    ws_public: str = Field(default="wss://stream-testnet.bybit.com/v5/public/linear")
    ws_private: str = Field(default="wss://stream-testnet.bybit.com/v5/private")


class TelegramSettings(BaseModel):
    enabled: bool = Field(default=True)
    bot_token: str = Field(default="", description="Telegram bot token")
    chat_id: Optional[str] = Field(default=None, description="Target chat ID")


class AppSettings(BaseModel):
    mode: str = Field(default="dev", description="dev | prod (dev=백테스트, prod=실거래)")
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT"])
    leverage: int = Field(default=3)
    margin_mode: str = Field(default="ISOLATED")
    order_size_usdt: float = Field(default=100.0, description="Notional per order")
    initial_balance_usdt: float = Field(default=10000.0, description="초기 계좌 잔고(개념상, 추적용)")
    backtest_limit: int = Field(default=500, description="백테스트 시 최신부터 가져올 캔들 개수")
    backtest_start: Optional[str] = Field(default=None, description="백테스트 시작시각 ISO (예: 2024-01-01T00:00:00)")
    backtest_end: Optional[str] = Field(default=None, description="백테스트 종료시각 ISO (예: 2024-01-15T00:00:00)")
    dry_run: bool = Field(default=True, description="If true, do not send live orders")
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    log_level: str = Field(default="INFO")


def load_settings(path: Optional[str] = None) -> AppSettings:
    """
    YAML 설정을 로드/검증한다. 환경변수 오버라이드는 사용하지 않는다.
    """
    config_data = {}
    if path:
        cfg_path = Path(path)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as fh:
                config_data = yaml.safe_load(fh) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    try:
        settings = AppSettings(**config_data)
        _apply_exchange_env_defaults(settings)
        return settings
    except ValidationError as exc:
        raise RuntimeError(f"Invalid configuration: {exc}") from exc


def deep_merge(a: dict, b: dict) -> dict:
    """얕은 재귀 병합: dict a를 기본으로 b를 덮어쓴 새 dict 반환."""
    result = dict(a or {})
    for key, val in (b or {}).items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            if val not in (None, "", {}):
                result[key] = val
            elif key not in result:
                result[key] = val
    return result


def _apply_exchange_env_defaults(settings: AppSettings) -> None:
    """exchange.env에 맞게 엔드포인트를 자동으로 맞춘다."""
    env = (settings.exchange.env or "").lower()
    if env == "testnet":
        settings.exchange.rest_base = "https://api-testnet.bybit.com"
        settings.exchange.ws_public = "wss://stream-testnet.bybit.com/v5/public/linear"
        settings.exchange.ws_private = "wss://stream-testnet.bybit.com/v5/private"
    elif env == "mainnet":
        settings.exchange.rest_base = "https://api.bybit.com"
        settings.exchange.ws_public = "wss://stream.bybit.com/v5/public/linear"
        settings.exchange.ws_private = "wss://stream.bybit.com/v5/private"
