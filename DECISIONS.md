# DECISIONS — sop-rag-agent

무인 작업 중 내린 판단과 이유를 한 줄씩 기록합니다.

- 문서: 실제 제조 SOP는 공개된 것이 거의 없고 사내 문서를 쓸 수 없어서, 한국어 합성 문서 22개를 직접 만들었다. 모든 문서 머리말과 README에 '합성'이라고 표시했다.
- 문서 속 판단 기준 수치는 factory-pdm(AI4I)의 고장 규칙과 맞췄다. 두 프로젝트가 '예측 → 조치'로 이어지게 하려고.
- 인젝션 방어를 검증하려고, 악성 지시문을 넣은 테스트 문서(NOTICE-001)를 일부러 코퍼스에 포함했다.
- 벡터DB는 Chroma 대신 FAISS를 골랐다. 문서 80청크 규모에서는 정확 검색(IndexFlatIP)으로 충분하고 의존성이 가볍기 때문이다.
- 임베딩은 `intfloat/multilingual-e5-small`을 썼다. 한국어를 지원하는 가벼운 로컬 모델이라 API 비용이 없다. e5 규칙대로 `query:`/`passage:` 접두어를 붙였다.
- BM25 토큰화는 형태소 분석기 대신 '단어 + 한글 글자 2-gram'으로 했다. Windows에서 설치 부담이 있는 Java 기반 분석기를 피하려고.
- 청킹은 고정 길이 대신 마크다운 섹션 경계로 했다. 인용 단위(문서 §섹션)와 맞추기 위해서다.
- 에이전트는 LangGraph(StateGraph + ToolNode + tools_condition)로 만들었다. `create_react_agent` 같은 고수준 헬퍼는 버전마다 위치·이름이 바뀌어 왔기 때문에, 그래프를 직접 구성해서 구조를 명확히 했다.
- LLM은 `langchain-anthropic` 1.7.10의 `ChatAnthropic`을 썼다. 설치한 버전의 소스 docstring과 공식 문서(docs.langchain.com, Anthropic 통합 페이지)에서 `effort`, `bind_tools`, ToolMessage 반환 방식을 확인했다.
- Haiku 5.5는 temperature 같은 샘플링 파라미터에 기본값이 아닌 값을 보내면 400을 반환하므로 temperature를 지정하지 않았다(재현성은 평가셋 반복 측정으로 대신 확인해야 함, 미측정).
- LLM 대체 경로는 Anthropic → Ollama(`qwen3:latest`) → 목업 순서다. Ollama 경로는 API 키를 비운 상태에서 실제로 답변이 나오는 것을 확인했다(약 85초/문항).
- 테스트(pytest)는 LLM을 호출하지 않도록 목업으로 바꿔 끼웠다. 비용과 네트워크 의존 없이 CI에서 돌게 하려고.
- 평가에서 '근거 일치도'는 Haiku를 판정자로 썼다. 자기 평가 편향이 있을 수 있어서 규칙 기반 지표(핵심 사실 포함률, 인용 정확도)를 함께 보고했다.
- 인젝션 판정: 처음에는 '악성 문구 출현'만 봤는데, 모델이 악성 문구를 인용하며 거부한 답변을 '따름'으로 잘못 세는 것을 답변 원문을 읽고 발견했다. '문구 출현 AND 거부 표현 없음'으로 규칙을 바꾸고, LLM을 다시 부르지 않고 저장된 답변으로 다시 판정했다(`--report-only`).
- 에이전트 trace에서 병렬 도구 호출 결과가 다른 호출에 붙는 버그가 있어서 `tool_call_id`로 짝을 맞추도록 고쳤다.
- factory-pdm 연동 주소는 `localhost` 대신 `127.0.0.1`을 기본값으로 했다. Windows에서 IPv6 우선 시도로 요청마다 약 2초가 늦어지는 것을 factory-pdm에서 확인했기 때문이다.
- Docker 이미지는 리눅스 기본 torch(CUDA 포함, 수 GB) 대신 CPU 전용 휠을 먼저 설치하도록 했다.
- Docker: `docker compose build`로 이미지를 만들고 api·ui 컨테이너를 띄워 /health, /ask, /agent(호스트의 factory-pdm 실시간 데이터 사용, mock=false)와 UI(200) 응답까지 확인했다.
- 응답 Content-Type에 `charset=utf-8`을 명시했다. 명시하지 않으면 Windows PowerShell 5.1의 Invoke-RestMethod가 한글 응답을 Latin-1로 읽어 깨지는 것을 직접 확인해서.
- README의 호출 예시는 curl 대신 `Invoke-RestMethod`로 썼다. PowerShell 5.1에서 `curl`은 Invoke-WebRequest의 별칭이라 curl 문법이 그대로 동작하지 않아서.
