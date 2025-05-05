"""Document processing functionality for the RAG system."""

import os
import glob
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag.config.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_CHROMA_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_KWARGS,
    EMBEDDING_ENCODE_KWARGS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(directory: str = DEFAULT_DATA_DIR, limit: int = 10) -> List[Document]:
    """Load and process PDF documents from the specified directory.
    
    Args:
        directory: Directory containing PDF files
        limit: Maximum number of files to process
        
    Returns:
        List of processed Document objects
    """
    documents = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    
    if limit:
        pdf_files = pdf_files[:limit]
    
    for pdf_file in pdf_files:
        try:
            # Load PDF as markdown
            loader = DoclingLoader(
                file_path=pdf_file,
                export_type=ExportType.MARKDOWN
            )
            docs = loader.load()
            
            # Configure markdown header splitter
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header_1"),
                    ("##", "Header_2"),
                    ("###", "Header_3")
                ]
            )
            
            # Split documents by headers
            split_docs = []
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_file)
                splits = header_splitter.split_text(doc.page_content)
                
                for split in splits:
                    split.metadata.update(doc.metadata)
                
                split_docs.extend(splits)
            
            documents.extend(split_docs)
            logger.info(f"Loaded and split {len(split_docs)} sections from {os.path.basename(pdf_file)}")
            
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")
    
    return documents

def create_vector_store(
    documents: List[Document],
    persist_directory: str = DEFAULT_CHROMA_DIR
) -> Chroma:
    """Create and return a vector store from the given documents.
    
    Args:
        documents: List of Document objects to store
        persist_directory: Directory to persist the vector store
        
    Returns:
        Chroma vector store instance
    """
    # Configure text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    
    # Configure embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

def load_vectorstore(persist_directory: str = DEFAULT_CHROMA_DIR) -> Chroma:
    """Load an existing vector store from the specified directory.
    
    Args:
        persist_directory: Directory containing the persisted vector store
        
    Returns:
        Chroma vector store instance
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore 