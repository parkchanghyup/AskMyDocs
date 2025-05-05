from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import os
from dotenv import load_dotenv

# Import RAG components
from rag.core import create_rag_graph, run_rag, load_vectorstore, load_documents, create_vector_store, delete_vectorstore
from rag.models import QueryRequest, QueryResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="건설안전지침 RAG API",
    description="건설 안전 지침에 관한 질문에 답변하는 RAG(Retrieval-Augmented Generation) API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG application on startup
@app.on_event("startup")
async def startup_event():
    # 기존 벡터 스토어 제거
    try:
        print("Checking for existing vector store...")
        delete_vectorstore()  # 기존 벡터 스토어 삭제
        print("Existing vector store deleted successfully!")
    except Exception as e:
        print(f"No existing vector store found or failed to delete: {e}")

    # 새로운 벡터 스토어 생성
    print("Creating new vector store...")
    documents = load_documents()
    app.state.vectorstore = create_vector_store(documents)
    print("Vector store created successfully!")

    app.state.rag_app = create_rag_graph(app.state.vectorstore)
    print("RAG application initialized!")

# Root endpoint - redirect to the HTML interface
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# API endpoints
@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Process a query using the RAG system and return the response
    """
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = run_rag(request.query, app.state.rag_app)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/api/config")
async def get_config():
    """
    Get configuration from environment variables
    """
    return {
        "system_title": os.getenv("SYSTEM_TITLE", "건설안전지침 RAG 시스템"),
        "initial_message": os.getenv("INITIAL_MESSAGE", "안녕하세요! 건설 안전 지침에 관한 질문이 있으시면 언제든지 물어보세요.")
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 