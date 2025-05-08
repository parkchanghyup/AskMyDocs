"""Constants and configuration values for the RAG system."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Model Configuration - separate models for different functions
GENERAL_MODEL_NAME = "models/" + os.environ.get("GEMINI_GENERAL_MODEL", "gemini-2.0-flash-lite")
RAG_MODEL_NAME = "models/" + os.environ.get("GEMINI_RAG_MODEL", "gemini-2.5-flash-preview-04-17")
MODEL_NAME = GENERAL_MODEL_NAME  # For backward compatibility

# Document Configuration
DOCUMENT_DOMAIN = os.getenv("DOCUMENT_DOMAIN", "일반 문서")
DEFAULT_DATA_DIR = "./data"
DEFAULT_CHROMA_DIR = "./chroma_db"

# Model Configuration
EMBEDDING_MODEL = "baai/bge-m3"
EMBEDDING_MODEL_KWARGS = {'device': 'cuda'}
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}

# Text Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 2048
TEMPERATURE = 0.1

# Search Configuration
TOP_K_RESULTS = 5 