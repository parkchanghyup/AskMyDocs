from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system
    """
    query: str = Field(..., description="The user's query to process")

class DocumentInfo(BaseModel):
    """
    Information about a document retrieved from the vector store
    """
    source: str = Field(..., description="The source of the document")

class QueryResponse(BaseModel):
    """
    Response model for the RAG system query
    """
    answer: str = Field(..., description="The generated answer to the query")
    thinking: str = Field(..., description="The reasoning process behind the answer")
    need_retrieval: bool = Field(..., description="Whether document retrieval was needed")
    documents: Optional[List[str]] = Field(None, description="List of document sources used for retrieval") 