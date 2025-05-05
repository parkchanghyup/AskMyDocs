"""Main entry point for the RAG system."""

import os
import shutil
import logging
from typing import Dict, Any

from rag.config.constants import DEFAULT_CHROMA_DIR
from rag.document_processor import load_documents, create_vector_store, load_vectorstore
from rag.rag_graph import create_rag_graph, run_rag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system class that coordinates all components."""
    
    def __init__(self, data_dir: str = "./data", chroma_dir: str = DEFAULT_CHROMA_DIR):
        """Initialize the RAG system.
        
        Args:
            data_dir: Directory containing PDF documents
            chroma_dir: Directory for the vector store
        """
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.rag_app = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system components."""
        try:
            # Try to load existing vector store
            vectorstore = load_vectorstore(self.chroma_dir)
            logger.info("Loaded existing vector store")
        except Exception as e:
            logger.info(f"Could not load existing vector store: {e}")
            logger.info("Creating new vector store from documents...")
            
            # Load and process documents
            documents = load_documents(self.data_dir)
            if not documents:
                raise ValueError(f"No documents found in {self.data_dir}")
            
            # Create new vector store
            vectorstore = create_vector_store(documents, self.chroma_dir)
            logger.info("Created new vector store")
        
        # Create RAG graph
        self.rag_app = create_rag_graph(vectorstore)
        logger.info("Initialized RAG system")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the RAG system.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.rag_app:
            raise RuntimeError("RAG system not initialized")
        
        return run_rag(query, self.rag_app)
    
    def reset_vectorstore(self):
        """Reset the vector store by deleting and reinitializing it."""
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)
            logger.info(f"Deleted vector store at {self.chroma_dir}")
        
        self._initialize()
        logger.info("Reset and reinitialized vector store")

def main():
    """Example usage of the RAG system."""
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example queries
    queries = [
        "안녕하세요",
        "주요 개념에 대해 알려주세요",
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag.process_query(query)
        print(f"Answer: {result['answer']}")
        print(f"Thinking: {result['thinking']}")
        if result['documents']:
            print(f"Sources: {result['documents']}")

if __name__ == "__main__":
    main() 