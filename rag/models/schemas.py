"""Pydantic schemas for the RAG API."""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for query API endpoint."""
    query: str = Field(..., description="User query string", examples=["건설 현장 안전 조치는 어떤 것들이 있나요?"])


class QueryResponse(BaseModel):
    """Response model for query API endpoint."""
    answer: str = Field(..., description="Generated answer to the query")
    thinking: str = Field(..., description="Reasoning process of the RAG system")
    need_retrieval: bool = Field(..., description="Whether document retrieval was needed")
    documents: Optional[List[str]] = Field(default=[], description="Source documents used for the answer") 