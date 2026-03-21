# AutoCoinMarginBot

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Bybit 암호화폐 자동 매매 봇. 앵커드 VWAP 전략을 기반으로 한 마진 트레이딩을 수행합니다.

## 🚀 주요 기능

- **앵커드 VWAP 전략**: 가격- VWAP 프리미엄/디스카운트를 기반으로 롱/숏 진입
- **부분 익절 (Partial Take Profit)**: TP1에서 30% 익절, 잔여 물량으로 TP2 추구
- **리스크 관리**: 최대 드로우다운 및 일일 손실 제한
- **시장 레짐 감지**: 추세/횡보 시장에 따라 전략 자동 전환
- **텔레그램 알림**: 실시간 거래 및 리스크 알림
- **백테스트 지원**: 과거 데이터로 전략 검증 및 파라미터 튜닝

## 📋 요구사항

- Python 3.11+
- Docker & Docker Compose (권장)
- Bybit API 키 (테스트넷 또는 메인넷)

## 🛠️ 설치 및 설정

### 1. 리포지토리 클론

```bash
git clone https://github.com/yourusername/AutoCoinMarginBot.git
cd AutoCoinMarginBot
```

### 2. 의존성 설치

**로컬 개발:**
```bash
pip install -r requirements.txt
# 또는
pip install .
```

**Docker 사용 (권장):**
```bash
docker compose build
```

### 3. 설정 파일 준비

`config.example.yaml`을 복사하여 `config.yaml` 생성:

```bash
cp config.example.yaml config.yaml
```

`config.yaml`을 편집하여 다음 설정:

- **API 키**: `exchange.api_key`, `exchange.api_secret`
- **환경**: `exchange.env` (testnet 또는 mainnet)
- **전략 파라미터**: TP/SL, 밴드 설정 등
- **텔레그램**: `telegram.bot_token`, `telegram.chat_id`

### 4. 실행

**개발 모드 (백테스트):**
```bash
python main.py
```

**프로덕션 모드 (실거래):**
```bash
docker compose up -d
```

**로그 확인:**
```bash
docker compose logs -f
```

## ⚙️ 설정 옵션

### 전략 파라미터

```yaml
strategy:
  # TP/SL 설정
  take_profit_move_long: 0.008    # 롱 TP1 (0.8%)
  take_profit_move_short: 0.008   # 숏 TP1
  stop_loss_long: 0.006          # 롱 SL (-0.6%)
  stop_loss_short: 0.006         # 숏 SL (+0.6%)

  # 부분 익절
  partial_take_profit_pct: 0.3    # TP1에서 30% 익절
  take_profit_move_long2: 0.016   # TP2 (TP1의 2배)

  # 리스크 관리
  max_drawdown_pct: 0.1           # 최대 드로우다운 (10%)
  daily_loss_pct: 0.05            # 일일 최대 손실 (5%)

  # VWAP 밴드
  band:
    k: 0.006                      # 프리미엄 임계값 (0.6%)
    candle_interval: 5m           # 캔들 간격
    vwap_lookback: 80             # VWAP 계산 기간

  # 시장 레짐
  regime:
    enabled: true
    neutral_behavior: vwap        # 중립 시장 동작
```

### 텔레그램 알림

거래 이벤트 및 리스크 경고를 텔레그램으로 받으려면:

1. [@BotFather](https://t.me/botfather)에서 봇 생성
2. 봇 토큰을 `telegram.bot_token`에 설정
3. 채팅 ID를 `telegram.chat_id`에 설정 (봇에게 메시지 보내고 `/getUpdates`로 확인)

## 📊 백테스트 및 튜닝

### 백테스트 실행

```bash
python main.py  # mode: dev
```

결과: `backtest_report.txt`, `backtest_equity_chart.png`

### 파라미터 튜닝

```bash
python tools/tune_backtest.py --k "0.005:0.01:0.001" --max-drawdown-pct "0.05:0.15:0.01" --out results.csv --apply-best
```

- 그리드 서치로 최적 파라미터 찾기
- `--apply-best`: 결과를 `config.yaml`에 자동 적용

## 🚀 클라우드 배포

### Docker Compose 사용

1. 서버에 프로젝트 업로드 (`.gitignore`된 파일 제외)
2. `config.yaml` 설정
3. 실행:

```bash
docker compose up -d
```

### 환경 변수

민감한 정보는 환경 변수로 관리:

```bash
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
export TELEGRAM_BOT_TOKEN="your_token"
```

`config.yaml`에서 `${BYBIT_API_KEY}` 등 사용.

## 📁 프로젝트 구조

```
AutoCoinMarginBot/
├── alerts/              # 텔레그램 알림
├── clients/             # Bybit API 클라이언트
├── config/              # 설정 관리
├── core/                # 메인 로직 (트레이더, 백테스터)
├── strategy/            # VWAP 전략
├── tools/               # 튜닝 도구
├── Dockerfile           # Docker 이미지
├── docker-compose.yml   # 컨테이너 구성
├── pyproject.toml       # 프로젝트 메타데이터
└── README.md
```

## 🔧 개발

### 코드 실행

```bash
# 가상환경 활성화
source .venv/bin/activate

# 테스트 실행
python -m pytest

# 코드 포맷팅
black .
isort .
```

### 기여

1. Fork 및 브랜치 생성
2. 변경사항 커밋
3. Pull Request 생성

## ⚠️ 주의사항

- **리스크**: 암호화폐 트레이딩은 고위험. 충분한 테스트 후 소액으로 시작
- **API 제한**: Bybit API rate limit 준수
- **보안**: API 키를 안전하게 관리. `.gitignore` 확인

## 📄 라이선스

MIT License - 자유롭게 사용 및 수정 가능

## 📞 지원

이슈나 질문은 GitHub Issues를 이용해주세요.
