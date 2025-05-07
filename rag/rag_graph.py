"""RAG graph functionality for the RAG system."""

import json
import logging
from typing import TypedDict, List, Annotated, Dict, Optional, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from rag.config.constants import GENERAL_MODEL_NAME, RAG_MODEL_NAME, TOP_K_RESULTS
from rag.llm_chain import query_analyzer_chain, general_chain, create_rag_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG state definition
class RAGState(TypedDict):
    query: str
    thinking: str
    answer: str
    messages: Annotated[List, add_messages]
    model: str
    documents: Optional[List[Document]]
    need_retrieval: bool

def analyze_query(state: RAGState) -> RAGState:
    """Analyze the query to determine if retrieval is needed.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated RAG state
    """
    query = state["query"]
    messages = [HumanMessage(content=query)]
    
    # Run query analysis
    result = query_analyzer_chain({"messages": messages})
    
    try:
        # Extract JSON from result
        if '{' in result and '}' in result:
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            json_str = result[start_idx:end_idx]
            
        analysis = json.loads(json_str)
        need_retrieval = analysis.get("need_retrieval", False)
        reasoning = analysis.get("reasoning", "")
    except json.JSONDecodeError:
        # Fallback for JSON parsing failure
        lower_query = query.lower()
        greeting_keywords = ["안녕", "반갑", "hello", "hi", "hey"]
        
        if any(keyword in lower_query for keyword in greeting_keywords):
            need_retrieval = False
            reasoning = "일반적인 인사로 판단되어 검색이 필요하지 않음"
        else:
            need_retrieval = True
            reasoning = "JSON 파싱 실패, 기본적으로 검색 수행"
    
    # Update state
    state["thinking"] = f"Need to search?: {need_retrieval} \n Reasoning: {reasoning}"
    state["need_retrieval"] = need_retrieval
    
    return state

def retrieve_documents(state: RAGState, vectorstore) -> RAGState:
    """Retrieve relevant documents from the vector store.
    
    Args:
        state: Current RAG state
        vectorstore: Vector store to search in
        
    Returns:
        Updated RAG state
    """
    query = state["query"]
    
    # Perform vector search
    docs = vectorstore.similarity_search(query, k=TOP_K_RESULTS)
    
    # Update state
    state["documents"] = docs
    state["thinking"] += f"\n검색된 문서 수: {len(docs)}"
    
    return state

def generate_rag_response(state: RAGState) -> RAGState:
    """Generate response using retrieved documents.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated RAG state
    """
    query = state["query"]
    docs = state["documents"]
    
    # Create document context
    context = "\n\n".join([
        f"문서: {doc.metadata.get('source', '알 수 없는 출처')}\n내용: {doc.page_content}"
        for doc in docs
    ])
    
    # Run RAG chain
    rag_chain = create_rag_chain()
    messages = [HumanMessage(content=query)]
    answer = rag_chain({"messages": messages, "context": context})
    
    # Update state
    state["answer"] = answer
    state["messages"].append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=answer))
    state["model"] = RAG_MODEL_NAME
    
    return state

def generate_general_response(state: RAGState) -> RAGState:
    """Generate general response without document retrieval.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated RAG state
    """
    query = state["query"]
    
    # Run general chain
    messages = [HumanMessage(content=query)]
    answer = general_chain({"messages": messages})
    
    # Update state
    state["answer"] = answer
    state["messages"].append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=answer))
    state["model"] = GENERAL_MODEL_NAME
    
    return state

def router(state: RAGState) -> str:
    """Route to next node based on retrieval need.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    return "retrieve" if state["need_retrieval"] else "general"

def create_rag_graph(vectorstore) -> StateGraph:
    """Create the RAG workflow graph.
    
    Args:
        vectorstore: Vector store to use for retrieval
        
    Returns:
        Compiled StateGraph
    """
    # Create retrieve function with vectorstore
    def retrieve_with_vectorstore(state):
        return retrieve_documents(state, vectorstore)
    
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_with_vectorstore)
    workflow.add_node("rag_response", generate_rag_response)
    workflow.add_node("general", generate_general_response)
    
    # Add edges
    workflow.add_conditional_edges(
        "analyze",
        router,
        {
            "retrieve": "retrieve",
            "general": "general"
        }
    )
    workflow.add_edge("retrieve", "rag_response")
    workflow.add_edge("rag_response", END)
    workflow.add_edge("general", END)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Compile graph
    return workflow.compile()

def run_rag(query: str, rag_app) -> Dict[str, Any]:
    """Run the RAG system with the given query.
    
    Args:
        query: User query
        rag_app: Compiled RAG graph
        
    Returns:
        Dictionary containing response and metadata
    """
    # Set initial state
    initial_state = {
        "query": query,
        "thinking": "",
        "answer": "",
        "messages": [],
        "model": "",  # Will be set based on whether retrieval is needed
        "documents": [],
        "need_retrieval": False
    }
    
    # Run graph
    result = rag_app.invoke(initial_state)
    
    # Set the model name in the result based on whether retrieval was needed
    if result["need_retrieval"]:
        result["model"] = RAG_MODEL_NAME
    else:
        result["model"] = GENERAL_MODEL_NAME
    
    # Return results
    return {
        "answer": result["answer"],
        "thinking": result["thinking"],
        "need_retrieval": result["need_retrieval"],
        "model": result["model"],
        "documents": [
            doc.metadata.get("source", "알 수 없는 출처")
            for doc in result.get("documents", [])
        ] if result["need_retrieval"] else []
    } 