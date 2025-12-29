"""Bybit REST 클라이언트(간단 스켈레톤).

v5 API 기준 레버리지/마진 설정, 티커 조회(VWAP 포함), 시장가 주문/취소를 지원한다.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from hashlib import sha256
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger

from config import AppSettings


class BybitClient:
    """Bybit v5 REST 호출용 간단 래퍼."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(base_url=settings.exchange.rest_base, timeout=10.0)
        self._vwap_cache: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
        self._ema_cache: Dict[Tuple[str, str, int, int, int], Tuple[float, Tuple[float, float]]] = {}
        self._lot_size_cache: Dict[str, Tuple[str, str]] = {}

    async def close(self) -> None:
        """HTTP 세션을 닫는다."""
        await self._client.aclose()

    def _auth_headers(
        self, body: Optional[dict] = None, query: Optional[dict] = None, raw_body: Optional[str] = None
    ) -> Dict[str, str]:
        """서명 포함 인증 헤더를 생성한다."""
        if not self.settings.exchange.api_key or not self.settings.exchange.api_secret:
            return {}

        api_key = self.settings.exchange.api_key
        secret = self.settings.exchange.api_secret
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        if raw_body is None:
            raw_body = self._json_dumps(body) if body else ""
        raw_query = httpx.QueryParams(query or {})
        payload = raw_body or str(raw_query)
        sign_payload = f"{timestamp}{api_key}{recv_window}{payload}".encode()
        signature = hmac.new(secret.encode(), sign_payload, sha256).hexdigest()

        return {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """티커 조회(VWAP/avgPrice 포함)."""
        endpoint = "/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        res = await self._client.get(endpoint, params=params)
        res.raise_for_status()
        data = res.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {data}")
        result = (data.get("result", {}).get("list") or [])
        return result[0] if result else {}

    async def get_tickers(self, symbols: list[str]) -> Dict[str, Dict[str, Any]]:
        """여러 심볼의 티커를 한 번에 조회한다."""
        if not symbols:
            return {}
        endpoint = "/v5/market/tickers"
        params = {"category": "linear"}
        if len(symbols) == 1:
            params["symbol"] = symbols[0]
        res = await self._client.get(endpoint, params=params)
        res.raise_for_status()
        data = res.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {data}")
        result = (data.get("result", {}).get("list") or [])
        target = set(symbols)
        tickers: Dict[str, Dict[str, Any]] = {}
        for item in result:
            symbol = item.get("symbol")
            if symbol in target:
                tickers[symbol] = item
        return tickers

    async def get_local_vwap(
        self, symbol: str, interval: str, limit: int, refresh_sec: int = 30
    ) -> Optional[float]:
        """캔들 기반 로컬 VWAP을 계산한다(캐시 사용)."""
        key = (symbol, interval, limit)
        now = time.time()
        cached = self._vwap_cache.get(key)
        if cached and (now - cached[0] < refresh_sec):
            return cached[1]

        endpoint = "/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": self._normalize_interval(interval),
            "limit": limit,
        }
        try:
            res = await self._client.get(endpoint, params=params)
            res.raise_for_status()
            data = res.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {data}")
        except Exception as exc:
            if cached:
                logger.warning(
                    "VWAP fetch failed; using cached value symbol={} error={}",
                    symbol,
                    exc,
                )
                return cached[1]
            logger.error("VWAP fetch failed symbol={} error={}", symbol, exc)
            return None

        klines = data.get("result", {}).get("list") or []
        sum_volume = 0.0
        sum_turnover = 0.0
        for k in klines:
            try:
                volume = float(k[5])
                turnover = float(k[6])
            except (TypeError, ValueError, IndexError):
                continue
            if volume <= 0 or turnover <= 0:
                continue
            sum_volume += volume
            sum_turnover += turnover

        if sum_volume <= 0:
            return None

        vwap = sum_turnover / sum_volume
        self._vwap_cache[key] = (now, vwap)
        return vwap

    async def get_ema_pair(
        self,
        symbol: str,
        interval: str,
        fast: int,
        slow: int,
        lookback: int,
        refresh_sec: int = 300,
    ) -> Optional[Tuple[float, float]]:
        """EMA fast/slow 값을 계산한다(캐시 사용)."""
        if fast <= 0 or slow <= 0 or lookback <= 0:
            return None
        fast_len, slow_len = (fast, slow) if fast <= slow else (slow, fast)
        key = (symbol, interval, fast_len, slow_len, lookback)
        now = time.time()
        cached = self._ema_cache.get(key)
        if cached and (now - cached[0] < refresh_sec):
            ema_fast, ema_slow = cached[1]
            return (ema_fast, ema_slow) if fast <= slow else (ema_slow, ema_fast)

        endpoint = "/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": self._normalize_interval(interval),
            "limit": lookback,
        }
        try:
            res = await self._client.get(endpoint, params=params)
            res.raise_for_status()
            data = res.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {data}")
        except Exception as exc:
            if cached:
                logger.warning(
                    "EMA fetch failed; using cached value symbol={} error={}",
                    symbol,
                    exc,
                )
                ema_fast, ema_slow = cached[1]
                return (ema_fast, ema_slow) if fast <= slow else (ema_slow, ema_fast)
            logger.error("EMA fetch failed symbol={} error={}", symbol, exc)
            return None

        klines = data.get("result", {}).get("list") or []
        parsed = []
        for k in klines:
            try:
                ts = int(k[0])
                close = float(k[4])
            except (TypeError, ValueError, IndexError):
                continue
            parsed.append((ts, close))
        if not parsed:
            return None
        parsed.sort(key=lambda x: x[0])
        closes = [c for _, c in parsed]
        ema_fast_val = self._calc_ema(closes, fast_len)
        ema_slow_val = self._calc_ema(closes, slow_len)
        if ema_fast_val is None or ema_slow_val is None:
            return None
        self._ema_cache[key] = (now, (ema_fast_val, ema_slow_val))
        return (ema_fast_val, ema_slow_val) if fast <= slow else (ema_slow_val, ema_fast_val)

    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """지갑 잔고를 조회한다."""
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": account_type}
        headers = self._auth_headers(query=params)
        res = await self._client.get(endpoint, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {data}")
        result = (data.get("result", {}).get("list") or [])
        return result[0] if result else {}

    async def get_positions(self, symbol: Optional[str] = None) -> list[Dict[str, Any]]:
        """보유 포지션을 조회한다."""
        endpoint = "/v5/position/list"
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        headers = self._auth_headers(query=params)
        res = await self._client.get(endpoint, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {data}")
        return data.get("result", {}).get("list") or []

    async def get_lot_size_filter(self, symbol: str) -> Optional[Tuple[str, str]]:
        """심볼의 최소 주문 수량/스텝을 조회한다."""
        cached = self._lot_size_cache.get(symbol)
        if cached:
            return cached

        endpoint = "/v5/market/instruments-info"
        params = {"category": "linear", "symbol": symbol}
        try:
            res = await self._client.get(endpoint, params=params)
            res.raise_for_status()
            data = res.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {data}")
        except Exception as exc:
            logger.error("Lot size fetch failed symbol={} error={}", symbol, exc)
            return None

        info_list = data.get("result", {}).get("list") or []
        if not info_list:
            return None
        lot = info_list[0].get("lotSizeFilter") or {}
        min_qty = lot.get("minOrderQty")
        step = lot.get("qtyStep")
        if not min_qty or not step:
            return None
        self._lot_size_cache[symbol] = (str(min_qty), str(step))
        return self._lot_size_cache[symbol]

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        """Bybit kline interval 포맷으로 변환."""
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

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """레버리지를 설정한다."""
        endpoint = "/v5/position/set-leverage"
        body = {"category": "linear", "symbol": symbol, "buyLeverage": str(leverage), "sellLeverage": str(leverage)}
        raw_body = self._json_dumps(body)
        headers = self._auth_headers(raw_body=raw_body)
        res = await self._client.post(endpoint, headers=headers, content=raw_body)
        res.raise_for_status()
        logger.info(f"레버리지 설정 완료 {symbol} 레버리지={leverage} (ret={res.json().get('retMsg')})")

    async def set_margin_mode(self, symbol: str, mode: str = "ISOLATED") -> None:
        """마진 모드(격리/교차)를 설정한다."""
        endpoint = "/v5/position/switch-isolated"
        body = {"category": "linear", "symbol": symbol, "tradeMode": 0 if mode.upper() == "CROSS" else 1}
        raw_body = self._json_dumps(body)
        headers = self._auth_headers(raw_body=raw_body)
        res = await self._client.post(endpoint, headers=headers, content=raw_body)
        res.raise_for_status()
        msg = res.json().get("retMsg")
        if msg == "unified account is forbidden":
            logger.warning("Unified 계정에서는 마진 모드 변경이 지원되지 않습니다. 스킵합니다.")
            return
        logger.info(f"마진 모드 설정 완료 {symbol} 모드={mode} (ret={msg})")

    async def place_market_order(
        self, symbol: str, side: str, qty: float, reduce_only: bool = False, position_idx: int = 0
    ) -> Dict[str, Any]:
        """시장가 주문을 전송한다."""
        endpoint = "/v5/order/create"
        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "ImmediateOrCancel",
            "reduceOnly": reduce_only,
            "positionIdx": position_idx,
        }
        raw_body = self._json_dumps(body)
        headers = self._auth_headers(raw_body=raw_body)
        res = await self._client.post(endpoint, headers=headers, content=raw_body)
        res.raise_for_status()
        data = res.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Order error {data}")
        return data

    async def cancel_all(self, symbol: str) -> None:
        """심볼의 모든 주문을 취소한다."""
        endpoint = "/v5/order/cancel-all"
        body = {"category": "linear", "symbol": symbol}
        raw_body = self._json_dumps(body)
        headers = self._auth_headers(raw_body=raw_body)
        res = await self._client.post(endpoint, headers=headers, content=raw_body)
        res.raise_for_status()
        logger.info(f"모든 주문 취소 {symbol} -> {res.json().get('retMsg')}")

    @staticmethod
    def _json_dumps(payload: Optional[dict]) -> str:
        """Bybit 서명과 동일한 JSON 문자열을 만든다."""
        return json.dumps(payload or {}, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _calc_ema(closes: list[float], period: int) -> Optional[float]:
        """단순 EMA 계산(가장 최신 EMA 반환)."""
        if period <= 0 or len(closes) < period:
            return None
        sma = sum(closes[:period]) / period
        ema = sma
        alpha = 2 / (period + 1)
        for price in closes[period:]:
            ema = (price - ema) * alpha + ema
        return ema


async def main_debug() -> None:
    """간단 수동 테스트 엔트리."""
    from config import load_settings

    settings = load_settings()
    client = BybitClient(settings)
    try:
        ticker = await client.get_ticker(settings.symbols[0])
        logger.info(f"샘플 티커: {ticker}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main_debug())
