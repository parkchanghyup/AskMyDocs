"""LLM chain functionality for the RAG system."""

from typing import Dict, Any
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

from rag.config.constants import (
    API_KEY,
    BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
)
from rag.config.prompts import (
    QUERY_ANALYZER_PROMPT,
    RAG_PROMPT,
    GENERAL_PROMPT,
)

# Initialize OpenAI client (using Gemini API)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def create_gemini_chain(system_prompt: str):
    """Create a Gemini model chain with the given system prompt.
    
    Args:
        system_prompt: System prompt to use for the chain
        
    Returns:
        Function that processes messages and returns model response
    """
    def call_gemini(input_dict: Dict[str, Any]) -> str:
        messages = input_dict.get("messages", [])
        formatted_messages = []
        
        # Add system message
        formatted_messages.append({"role": "system", "content": system_prompt})
        
        # Add user and assistant messages
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
        
        # Call API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    return call_gemini

def create_rag_chain():
    """Create a RAG chain that uses context from retrieved documents.
    
    Returns:
        Function that processes messages and context to return model response
    """
    def call_gemini_with_context(input_dict: Dict[str, Any]) -> str:
        messages = input_dict.get("messages", [])
        context = input_dict.get("context", "")
        
        # Add context to system message
        system_content = RAG_PROMPT.replace("{context}", context)
        formatted_messages = []
        
        # Add system message
        formatted_messages.append({"role": "system", "content": system_content})
        
        # Add user and assistant messages
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
        
        # Call API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    return call_gemini_with_context

# Initialize chains
query_analyzer_chain = create_gemini_chain(QUERY_ANALYZER_PROMPT)
general_chain = create_gemini_chain(GENERAL_PROMPT) 