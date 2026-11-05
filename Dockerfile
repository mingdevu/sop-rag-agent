FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# 리눅스 기본 torch 휠은 CUDA 포함(수 GB)이라, CPU 전용 휠을 먼저 설치해 이미지 크기를 줄인다
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
# 임베딩 모델 다운로드 + 색인 생성을 빌드 시점에 끝내서 첫 요청 지연을 없앤다
RUN python -m sop_rag.retriever
EXPOSE 8100 8501
# API 키는 이미지에 넣지 않고 실행 시 env_file(.env)로 주입한다
CMD ["uvicorn", "sop_rag.api:app", "--host", "0.0.0.0", "--port", "8100"]
