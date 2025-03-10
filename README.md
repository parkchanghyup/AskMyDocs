# Ask-My-Docs

이 프로젝트는 Documents에 기반하여 답변하는 RAG API를 FastAPI를 사용하여 구현한 것입니다.

## Project Architecture

![아키텍쳐](./image/img.png)

## 기능

- PDF, txt, word 등의 문서를 벡터 데이터베이스에 저장
- 사용자 질문에 대한 의도 분석 (검색 필요 여부 판단)
- 관련 문서 검색 및 컨텍스트 기반 응답 생성
- 웹 인터페이스를 통한 대화형 상호작용

## 기술 스택

- **FastAPI**: 웹 API 프레임워크
- **LangChain**: RAG 파이프라인 구성
- **LangGraph**: 워크플로우 관리
- **Gemini API**: 대규모 언어 모델
- **ChromaDB**: 벡터 데이터베이스
- **HuggingFace Embeddings**: 한국어 임베딩 모델

## 설치 방법

1. 저장소 클론:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. 가상 환경 생성 및 활성화:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정:
   `.env` 파일을 생성하고 다음 내용을 추가합니다:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL_NAME=gemini-2.0-flash-001
   SYSTEM_TITLE=메인 페이지의 title
   INITIAL_MESSAGE:대화 창의 초기 메시지
   ```

## 실행 방법

1. 서버 실행:
   ```bash
   python app.py
   ```

2. 웹 브라우저에서 접속:
   ```
   http://localhost:8000
   ```

## API 엔드포인트

- `GET /`: 웹 인터페이스로 리다이렉트
- `POST /api/query`: RAG 시스템에 질문 전송
  - 요청 본문: `{"query": "질문 내용"}`
  - 응답: `{"answer": "응답", "thinking": "사고 과정", "need_retrieval": true/false, "documents": ["참고 문서 목록"]}`
- `GET /api/health`: 서버 상태 확인

## 프로젝트 구조

```
.
├── app.py                  # FastAPI 애플리케이션 진입점
├── rag/                    # RAG 시스템 모듈
│   ├── __init__.py
│   ├── core.py             # RAG 핵심 기능 구현
│   ├── models/             # 데이터 모델
│   │   ├── __init__.py
│   │   └── schemas.py      # Pydantic 스키마
│   └── utils/              # 유틸리티 함수
│       ├── __init__.py
│       └── logging_utils.py # 로깅 유틸리티
├── static/                 # 정적 파일
│   └── index.html          # 웹 인터페이스
├── data/                   # 데이터 디렉토리
│   └── pdf/                # vector db로 변환할 데이터
├── chroma_db/              # 벡터 데이터베이스 저장소
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 문서
```
