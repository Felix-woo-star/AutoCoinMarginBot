flowchart TD
    A[시작] --> B[CLI 인자 파싱<br/>--config/--iterations/--live]
    B --> C[config.yaml 로드 -> AppSettings]
    C --> D[TelegramAlerter 초기화]
    D --> E[텔레그램 시작 알림 전송]
    E --> F{mode == "prod"?}

    %% dev 경로
    F -- dev --> G[Backtester 생성]
    G --> H[Bybit REST kline 조회\n(backtest_limit/start/end 반영)]
    H --> I[캔들 순회하며 AnchoredVWAPStrategy.evaluate]
    I --> J[LONG/SHORT 시그널에 따라 가상 포지션 관리<br/>TP/SL/부분익절]
    J --> K[성과 집계(승률/PnL/MDD 등)]
    K --> L{trades 존재?}
    L -- 예 --> M[텔레그램 백테스트 요약 전송]
    L -- 아니오 --> N[경고 로그 '체결 없음']
    M & N --> Z[종료(alerter.close)]

    %% prod 경로
    F -- prod --> O[AnchoredVWAPStrategy + BybitClient + Trader 생성]
    O --> P{dry_run? (--live 여부)}
    P -- 예(true) --> Q[모든 주문 관련 API는 시뮬레이션만 수행]
    P -- 아니오(false) --> R[초기 레버리지/마진 모드 설정 후 실주문 가능]
    Q --> S
    R --> S[iterations <=0 ? -> 무한 루프 / >0 ? 설정 횟수 반복]
    S --> T[각 루프에서 모든 심볼에 대해 get_ticker 호출]
    T --> U[Trader.handle_ticker(ticker)]
    U --> V{포지션 상태}
    V -- 미보유 --> W{시그널?}
        W -- LONG_ENTRY --> X[calc qty -> _open_position LONG]
        W -- SHORT_ENTRY --> Y[calc qty -> _open_position SHORT]
        W -- None --> T
    V -- LONG 보유 --> AA{시그널?}
        AA -- SHORT_ENTRY --> BB[_close_position (reason Opposite)]
        AA -- 기타 --> CC[_manage_long: TP/SL 체크]
    V -- SHORT 보유 --> DD{시그널?}
        DD -- LONG_ENTRY --> EE[_close_position (reason Opposite)]
        DD -- 기타 --> FF[_manage_short: TP/SL 체크]

    %% 주문/알림 세부
    X --> GG{dry_run?}
    Y --> GG
    CC --> GG
    FF --> GG
    BB --> GG
    EE --> GG
    GG -- true --> HH[시장가 주문 미전송, 로컬 잔고/알림만 갱신]
    GG -- false --> II[BybitClient.place_market_order\n(set_leverage/cancel reduceOnly 포함)]
    HH & II --> JJ[텔레그램에 진입/부분익절/청산 메시지 발송]
    JJ --> KK[루프 계속: asyncio.sleep(1)]
    KK --> T

    BB & EE --> LL[포지션 리셋, 잔고 로그]
    CC & FF --> MM[조건 충족 시 _partial_take_profit]
    MM --> JJ

    %% 종료
    Z & LL --> NN[BybitClient.close + TelegramAlerter.close]
