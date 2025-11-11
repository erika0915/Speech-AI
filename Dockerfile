FROM python:3.13-slim

WORKDIR /app

# 시스템 의존성 설치 
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \        
       libsndfile1 \  
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
# requirements.txt만 먼저 복사하여 설치
COPY requirements.txt .

# CPU 전용 PyTorch를 명시
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 모델 어댑터 복사 
COPY ./models ./models

COPY ./app ./app

EXPOSE 8000

# 서버 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]