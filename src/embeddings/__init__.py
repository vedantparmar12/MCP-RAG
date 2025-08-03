"""
Embedding providers for the Crawl4AI RAG MCP server.
"""
from .base import BaseEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .cohere_provider import CohereEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider
from .manager import EmbeddingManager

__all__ = [
    'BaseEmbeddingProvider',
    'OpenAIEmbeddingProvider',
    'CohereEmbeddingProvider',
    'OllamaEmbeddingProvider',
    'HuggingFaceEmbeddingProvider',
    'EmbeddingManager'
]