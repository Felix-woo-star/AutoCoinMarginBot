FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 프로젝트 메타 파일 복사
COPY pyproject.toml README.md ./

# 소스 복사
COPY alerts alerts
COPY clients clients
COPY config config
COPY core core
COPY strategy strategy
COPY __init__.py main.py ./

# 의존성 설치 (pyproject 기반)
RUN pip install --no-cache-dir .

# 기본 실행: config.yaml 자동 로드
CMD ["python", "main.py"]
