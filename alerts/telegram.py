"""텔레그램 알림 송신기.

bot_token/chat_id가 설정되어 있으면 실행 이벤트(백테스트 요약, 실거래 진입/청산 등)를 전송한다.
"""

from __future__ import annotations

import httpx
from loguru import logger

from config import TelegramSettings


class TelegramAlerter:
    """텔레그램 봇으로 메시지를 전송한다."""

    def __init__(self, settings: TelegramSettings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        """HTTP 클라이언트를 정리한다."""
        await self._client.aclose()

    async def send(self, message: str) -> None:
        """텔레그램으로 알림을 전송한다."""
        if not self.settings.enabled:
            logger.debug(f"[텔레그램] 비활성화로 알림 건너뜀: {message}")
            return
        if not self.settings.bot_token or not self.settings.chat_id:
            logger.warning("[텔레그램] bot_token/chat_id 미설정으로 전송 불가")
            return
        url = f"https://api.telegram.org/bot{self.settings.bot_token}/sendMessage"
        payload = {"chat_id": self.settings.chat_id, "text": message}
        try:
            logger.debug(f"[텔레그램] 전송 시도: chat_id={self.settings.chat_id}")
            res = await self._client.post(url, json=payload)
            res.raise_for_status()
        except Exception as exc:  # pragma: no cover - thin wrapper
            body = None
            try:
                if "res" in locals():
                    body = await res.text()
            except Exception:
                body = None
            logger.error(f"[텔레그램] 알림 전송 실패: {exc} | 응답={body}")
        else:
            logger.info("[텔레그램] 알림 전송 완료")
