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
