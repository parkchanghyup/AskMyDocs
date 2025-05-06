from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import os
from dotenv import load_dotenv

# Import RAG components from the new modular structure
from rag import RAGSystem
from rag.models.schemas import QueryRequest, QueryResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="문서와 관련된 질문에 답변하는 RAG(Retrieval-Augmented Generation) API",
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

# Initialize RAG system on startup
@app.on_event("startup")
async def startup_event():
    print("Initializing RAG system...")
    # Reset vector store on startup
    app.state.rag_system = RAGSystem()
    app.state.rag_system.reset_vectorstore()
    print("RAG system initialized!")

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
        result = app.state.rag_system.process_query(request.query)
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
        "system_title": os.getenv("SYSTEM_TITLE", "Please question about docs"),
        "initial_message": os.getenv("INITIAL_MESSAGE", "Hello.")
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 