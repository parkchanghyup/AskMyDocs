import warnings
warnings.filterwarnings('ignore')

from typing import TypedDict, List, Annotated, Dict, Optional, Any
import logging
import os
import json
import glob
from dotenv import load_dotenv
from openai import OpenAI
from dotenv import load_dotenv



# LangChain imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 로드
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBwX8l84Jm5G1uD0TiX3GT8iI47BgqrLYU")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "models/" + os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash-001")

# OpenAI 클라이언트 초기화 (Gemini API 사용)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# RAG 상태 정의
class RAGState(TypedDict):
    query: str
    thinking: str
    answer: str
    messages: Annotated[List, add_messages]
    model: str
    documents: Optional[List[Document]]
    need_retrieval: bool

# 문서 로딩 함수
def load_documents(directory="./data", limit=10):
    """PDF 문서들을 로드하는 함수 (DoclingLoader 사용)"""
    documents = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    
    # 처리 속도를 위해 일부 파일만 사용 (필요시 전체 파일 사용)
    if limit:
        pdf_files = pdf_files[:limit]
    
    for pdf_file in pdf_files:
        try:
            # DoclingLoader를 사용하여 PDF를 마크다운으로 변환하여 로드
            loader = DoclingLoader(
                file_path=pdf_file,
                export_type=ExportType.MARKDOWN
            )
            docs = loader.load()
            
            # 마크다운 헤더 기반 텍스트 분할기 설정
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header_1"),
                    ("##", "Header_2"),
                    ("###", "Header_3")
                ]
            )
            
            # 각 문서를 헤더 기반으로 분할
            split_docs = []
            for doc in docs:
                # 파일 이름을 메타데이터에 추가
                doc.metadata["source"] = os.path.basename(pdf_file)
                
                # 헤더 기반 분할 적용
                splits = header_splitter.split_text(doc.page_content)
                
                # 분할된 각 청크에 원본 메타데이터 추가
                for split in splits:
                    split.metadata.update(doc.metadata)
                
                split_docs.extend(splits)
            
            documents.extend(split_docs)
            print(f"Loaded and split {len(split_docs)} sections from {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    return documents

# 벡터 스토어 생성 함수
def create_vector_store(documents, persist_directory="./chroma_db"):
    """문서를 벡터 스토어에 저장하는 함수"""
    # 문서가 이미 헤더 기반으로 분할되어 있으므로, 추가 분할이 필요 없는 경우 주석 처리
    # 필요시 추가 분할을 위해 코드 유지
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # 문서가 이미 충분히 분할되어 있다면 아래 라인을 주석 처리하고 chunks = documents 사용
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")
    
    # 임베딩 모델 설정 (한국어에 적합한 모델 사용)
    embeddings = HuggingFaceEmbeddings(
        model_name="baai/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 벡터 스토어 생성
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

# 기존 벡터 스토어 로드 함수
def load_vectorstore(persist_directory="./chroma_db"):
    """기존 벡터 스토어를 로드하는 함수"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

# Gemini 체인 생성 함수
def create_gemini_chain(system_prompt):
    """Gemini 모델을 사용하는 체인 생성"""
    def call_gemini(input_dict):
        messages = input_dict.get("messages", [])
        formatted_messages = []
        
        # 시스템 메시지 추가
        formatted_messages.append({"role": "system", "content": system_prompt})
        
        # 사용자 메시지 추가
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
        
        # API 호출
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            temperature=0.1,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
    
    return call_gemini

document_domain = os.getenv("DOCUMENT_DOMAIN", "일반 문서")

# 쿼리 분석 프롬프트 정의
query_analyzer_prompt = """
당신은 사용자의 질문을 분석하여 외부 정보 검색이 필요한지 판단하는 전문가입니다.
질문이 {document_domain}와 관련된 구체적인 사실, 데이터, 또는 전문 지식을 요구하는 경우, 검색이 필요하다고 판단하세요.
일반적인 대화, 인사, 또는 {document_domain}와 관련 없는 질문은 검색이 필요하지 않습니다.

반드시 다음 형식의 유효한 JSON만 응답하세요. 다른 텍스트는 포함하지 마세요:
{{
  "need_retrieval": true/false,
  "reasoning": "판단 이유에 대한 간략한 설명"
}}

예시 1 - 인사:
입력: "안녕하세요, 반갑습니다."
출력: {{"need_retrieval": false, "reasoning": "일반적인 인사이므로 검색이 필요하지 않음"}}

예시 2 - 도메인 관련 질문:
입력: "주요 개념에 대해 알려주세요."
출력: {{"need_retrieval": true, "reasoning": "{document_domain}와 관련된 구체적인 정보를 요구하므로 검색 필요"}}
"""
formatted_prompt = query_analyzer_prompt.format(document_domain=document_domain)

# RAG 프롬프트
rag_prompt = """
당신은 다양한 주제에 대해 문서 정보를 바탕으로 답변하는 전문가입니다. 사용자의 질문에 대해 제공된 문서 정보를 활용하여 정확하고 유용한 답변을 제공하세요.
문서 정보에 답변이 없는 경우, 솔직하게 모른다고 말하고 관련 주제에 대한 일반적인 지식이나 원칙에 기반한 조언을 제공하세요.
답변은 항상 한국어로 제공하세요.

참고 문서:
{context}
"""

# 일반 응답 프롬프트
general_prompt = """
당신은 다양한 주제에 대한 일반적인 지식을 가진 도우미입니다. 사용자의 질문에 친절하고 유용한 답변을 제공하세요.
구체적인 사실이나 전문적 데이터가 필요한 경우, 정확한 정보가 부족할 수 있으므로 관련 주제의 일반적인 원칙이나 상식에 기반한 조언을 제공하세요.
답변은 항상 한국어로 제공하세요.
"""

# 체인 초기화
query_analyzer_chain = create_gemini_chain(query_analyzer_prompt)

# RAG 체인 초기화
def create_rag_chain():
    """RAG 체인 생성"""
    def call_gemini_with_context(input_dict):
        messages = input_dict.get("messages", [])
        context = input_dict.get("context", "")
        
        # 시스템 메시지에 컨텍스트 추가
        system_content = rag_prompt.replace("{context}", context)
        formatted_messages = []
        
        # 시스템 메시지 추가
        formatted_messages.append({"role": "system", "content": system_content})
        
        # 사용자 메시지 추가
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
        
        # API 호출
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            temperature=0.1,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
    
    return call_gemini_with_context

# 일반 체인 초기화
general_chain = create_gemini_chain(general_prompt)

# LangGraph 노드 함수 구현
def analyze_query(state: RAGState) -> RAGState:
    """쿼리를 분석하여 검색이 필요한지 판단하는 노드"""
    query = state["query"]
    messages = [HumanMessage(content=query)]
    
    # 쿼리 분석 실행
    result = query_analyzer_chain({"messages": messages})
    
    try:
        # JSON 파싱
        # 결과에서 JSON 부분만 추출 시도
        json_str = result
        # 중괄호로 시작하는 부분 찾기
        if '{' in result and '}' in result:
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            json_str = result[start_idx:end_idx]
            
        analysis = json.loads(json_str)
        need_retrieval = analysis.get("need_retrieval", False)
        reasoning = analysis.get("reasoning", "")
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 기본값 사용
        # 인사말이나 간단한 대화인지 확인
        lower_query = query.lower()
        greeting_keywords = ["안녕", "반갑", "hello", "hi", "hey"]
        
        # 인사말인 경우 검색 불필요로 설정
        if any(keyword in lower_query for keyword in greeting_keywords):
            need_retrieval = False
            reasoning = "일반적인 인사로 판단되어 검색이 필요하지 않음"
        else:
            need_retrieval = True  # 기본적으로 검색 수행
            reasoning = "JSON 파싱 실패, 기본적으로 검색 수행"
    
    # 상태 업데이트
    state["thinking"] = f"검색 필요 여부: {need_retrieval}, 이유: {reasoning}"
    state["need_retrieval"] = need_retrieval
    
    return state

def retrieve_documents(state: RAGState, vectorstore) -> RAGState:
    """벡터 스토어에서 관련 문서를 검색하는 노드"""
    query = state["query"]
    
    # 벡터 검색 수행
    docs = vectorstore.similarity_search(query, k=5)
    
    # 상태 업데이트
    state["documents"] = docs
    state["thinking"] += f"\n검색된 문서 수: {len(docs)}"
    
    return state

def generate_rag_response(state: RAGState) -> RAGState:
    """검색 결과를 바탕으로 응답을 생성하는 노드"""
    query = state["query"]
    docs = state["documents"]
    
    # 문서 컨텍스트 생성
    context = "\n\n".join([f"문서: {doc.metadata.get('source', '알 수 없는 출처')}\n내용: {doc.page_content}" for doc in docs])
    
    # RAG 체인 실행
    rag_chain = create_rag_chain()
    messages = [HumanMessage(content=query)]
    answer = rag_chain({"messages": messages, "context": context})
    
    # 상태 업데이트
    state["answer"] = answer
    state["messages"].append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=answer))
    
    return state

def generate_general_response(state: RAGState) -> RAGState:
    """일반적인 응답을 생성하는 노드 (검색 없음)"""
    query = state["query"]
    
    # 일반 체인 실행
    messages = [HumanMessage(content=query)]
    answer = general_chain({"messages": messages})
    
    # 상태 업데이트
    state["answer"] = answer
    state["messages"].append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=answer))
    
    return state

# 라우터 함수 정의
def router(state: RAGState):
    """검색 필요 여부에 따라 다음 노드를 결정하는 라우터"""
    if state["need_retrieval"]:
        return "retrieve"
    else:
        return "general"

# LangGraph 그래프 생성 함수
def create_rag_graph(vectorstore):
    """RAG 그래프를 생성하는 함수"""
    # 벡터 스토어를 사용하는 retrieve 함수 생성
    def retrieve_with_vectorstore(state):
        return retrieve_documents(state, vectorstore)
    
    # 그래프 생성
    workflow = StateGraph(RAGState)
    
    # 노드 추가
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_with_vectorstore)
    workflow.add_node("rag_response", generate_rag_response)
    workflow.add_node("general", generate_general_response)
    
    # 라우터 노드 추가
    def route(state):
        return router(state)
    
    # 엣지 추가 
    workflow.add_conditional_edges(
        "analyze",
        route,
        {
            "retrieve": "retrieve",
            "general": "general"
        }
    )
    workflow.add_edge("retrieve", "rag_response")
    workflow.add_edge("rag_response", END)
    workflow.add_edge("general", END)
    
    # 시작 노드 설정
    workflow.set_entry_point("analyze")
    
    # 그래프 컴파일
    return workflow.compile()

# RAG 시스템 실행 함수
def run_rag(query: str, rag_app):
    """RAG 시스템을 실행하는 함수"""
    # 초기 상태 설정
    initial_state = {
        "query": query,
        "thinking": "",
        "answer": "",
        "messages": [],
        "model": MODEL_NAME,
        "documents": [],
        "need_retrieval": False
    }
    
    # 그래프 실행
    result = rag_app.invoke(initial_state)
    
    # 결과 반환
    return {
        "answer": result["answer"],
        "thinking": result["thinking"],
        "need_retrieval": result["need_retrieval"],
        "documents": [doc.metadata.get("source", "알 수 없는 출처") for doc in result.get("documents", [])] if result["need_retrieval"] else []
    } 