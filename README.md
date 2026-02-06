# AutoCoinMarginBot
bybit coin 자동매매 bot

## 설정 요약
- 설정 파일: `config.yaml`
- 설정만 바꿨다면 `docker compose restart`
- 코드까지 바꿨다면 `docker compose build` 후 `docker compose up -d`

## 부분익절 보호 + TP2 추천값 (TP1=0.5% 기준)
부분익절(TP1) 이후 잔여 물량 손절을 진입가(버퍼 포함)로 올리고, TP2로 추가 익절합니다.

```yaml
strategy:
  take_profit_move_long: 0.005
  take_profit_move_short: 0.005
  take_profit_move_long2: 0.0075
  take_profit_move_short2: 0.0075
  move_stop_to_entry_after_partial: true
  break_even_buffer_pct: 0.001
```

## 레짐 기반 전략(정교 버전)
시장 상태를 추세/횡보로 분류해 전략을 자동 전환합니다.

- 추세(trend): EMA 방향에 따라 추세 추종
- 횡보(range): VWAP 프리미엄/디스카운트 기준 진입
- 중립(neutral): `neutral_behavior`로 동작 선택

```yaml
strategy:
  regime:
    enabled: true
    interval: 1h
    ema_fast: 20
    ema_slow: 50
    atr_period: 14
    lookback: 200
    refresh_sec: 300
    ema_spread_trend_pct: 0.002
    atr_trend_pct: 0.003
    ema_spread_range_pct: 0.001
    atr_range_pct: 0.002
    neutral_behavior: vwap # vwap | trend | none
```

레짐 모드가 켜져 있으면 `trend_filter`는 중복 방지를 위해 무시됩니다.
