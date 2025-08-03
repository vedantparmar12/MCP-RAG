"""
Embedding manager for handling multiple embedding providers.
"""
import os
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio
from .base import BaseEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .cohere_provider import CohereEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .huggingface_provider import HuggingFaceEmbeddingProvider


class EmbeddingManager:
    """Manages multiple embedding providers with automatic selection and fallback."""
    
    def __init__(self, default_provider: str = "auto"):
        """
        Initialize the embedding manager.
        
        Args:
            default_provider: Default provider to use ('openai', 'cohere', 'ollama', 'huggingface', 'auto')
        """
        self.providers: Dict[str, BaseEmbeddingProvider] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.default_provider = default_provider
        self.current_provider: Optional[str] = None
        
        # Initialize providers based on available credentials
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available embedding providers based on environment variables."""
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                self.providers["openai"] = OpenAIEmbeddingProvider(model=model)
                print(f"Initialized OpenAI embeddings with model: {model}")
            except Exception as e:
                print(f"Failed to initialize OpenAI provider: {e}")
        
        # Cohere
        if os.getenv("COHERE_API_KEY"):
            try:
                model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
                use_v2 = os.getenv("COHERE_USE_V2", "false").lower() == "true"
                self.providers["cohere"] = CohereEmbeddingProvider(model=model, use_v2=use_v2)
                print(f"Initialized Cohere embeddings with model: {model} (v2: {use_v2})")
            except Exception as e:
                print(f"Failed to initialize Cohere provider: {e}")
        
        # Ollama (check if running)
        if os.getenv("ENABLE_OLLAMA", "false").lower() == "true":
            try:
                model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.providers["ollama"] = OllamaEmbeddingProvider(model=model, base_url=base_url)
                print(f"Initialized Ollama embeddings with model: {model}")
            except Exception as e:
                print(f"Failed to initialize Ollama provider: {e}")
        
        # HuggingFace
        if os.getenv("ENABLE_HUGGINGFACE", "false").lower() == "true":
            try:
                model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                use_api = os.getenv("HUGGINGFACE_USE_API", "false").lower() == "true"
                api_token = os.getenv("HUGGINGFACE_API_TOKEN") if use_api else None
                device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
                
                self.providers["huggingface"] = HuggingFaceEmbeddingProvider(
                    model=model,
                    use_api=use_api,
                    api_token=api_token,
                    device=device
                )
                print(f"Initialized HuggingFace embeddings with model: {model} (API: {use_api})")
            except Exception as e:
                print(f"Failed to initialize HuggingFace provider: {e}")
        
        if not self.providers:
            raise ValueError("No embedding providers could be initialized. Please check your environment variables.")
        
        # Set current provider
        if self.default_provider == "auto":
            self.current_provider = self._select_best_provider()
        elif self.default_provider in self.providers:
            self.current_provider = self.default_provider
        else:
            self.current_provider = list(self.providers.keys())[0]
        
        print(f"Using {self.current_provider} as the default embedding provider")
    
    def _select_best_provider(self) -> str:
        """
        Select the best provider based on performance history and availability.
        
        Returns:
            Name of the selected provider
        """
        if not self.performance_history:
            # No history, use priority order
            priority_order = ["openai", "cohere", "huggingface", "ollama"]
            for provider in priority_order:
                if provider in self.providers:
                    return provider
            return list(self.providers.keys())[0]
        
        # Calculate scores based on latency, success rate, and cost
        scores = {}
        for provider_name in self.providers:
            if provider_name in self.performance_history:
                history = self.performance_history[provider_name][-100:]  # Last 100 operations
                
                # Calculate metrics
                success_rate = sum(1 for h in history if h["success"]) / len(history)
                avg_latency = sum(h["latency"] for h in history if h["success"]) / max(1, sum(1 for h in history if h["success"]))
                cost_per_token = self.providers[provider_name].get_cost_per_token()
                
                # Score formula (higher is better)
                # Prioritize success rate, then latency, then cost
                score = (success_rate * 100) - (avg_latency * 10) - (cost_per_token * 1000000)
                scores[provider_name] = score
            else:
                # No history for this provider, give it a neutral score
                scores[provider_name] = 50.0
        
        # Return provider with highest score
        return max(scores, key=scores.get)
    
    def _record_performance(self, provider: str, success: bool, latency: float, tokens: int = 0):
        """Record performance metrics for a provider."""
        if provider not in self.performance_history:
            self.performance_history[provider] = []
        
        self.performance_history[provider].append({
            "timestamp": datetime.now(),
            "success": success,
            "latency": latency,
            "tokens": tokens
        })
        
        # Keep only last 1000 records per provider
        if len(self.performance_history[provider]) > 1000:
            self.performance_history[provider] = self.performance_history[provider][-1000:]
    
    async def create_embedding(self, text: str, provider: Optional[str] = None) -> List[float]:
        """
        Create an embedding using the specified or current provider.
        
        Args:
            text: Text to create an embedding for
            provider: Specific provider to use (optional)
            
        Returns:
            List of floats representing the embedding
        """
        provider_name = provider or self.current_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        start_time = time.time()
        try:
            embedding = await self.providers[provider_name].create_embedding(text)
            latency = time.time() - start_time
            
            # Validate embedding
            if self.providers[provider_name].validate_embedding(embedding):
                self._record_performance(provider_name, True, latency)
                return embedding
            else:
                raise ValueError("Invalid embedding returned")
                
        except Exception as e:
            latency = time.time() - start_time
            self._record_performance(provider_name, False, latency)
            print(f"Error with {provider_name} provider: {e}")
            
            # Try fallback providers
            for fallback in self.providers:
                if fallback != provider_name:
                    try:
                        print(f"Trying fallback provider: {fallback}")
                        embedding = await self.providers[fallback].create_embedding(text)
                        if self.providers[fallback].validate_embedding(embedding):
                            return embedding
                    except Exception as fallback_error:
                        print(f"Fallback {fallback} also failed: {fallback_error}")
            
            # All providers failed, return zero embedding
            print("All providers failed, returning zero embedding")
            return [0.0] * self.get_dimension()
    
    async def create_embeddings_batch(self, texts: List[str], provider: Optional[str] = None) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to create embeddings for
            provider: Specific provider to use (optional)
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        provider_name = provider or self.current_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        start_time = time.time()
        try:
            embeddings = await self.providers[provider_name].create_embeddings_batch(texts)
            latency = time.time() - start_time
            
            # Validate embeddings
            valid_count = sum(1 for emb in embeddings if self.providers[provider_name].validate_embedding(emb))
            
            if valid_count == len(embeddings):
                self._record_performance(provider_name, True, latency, len(texts))
                return embeddings
            else:
                print(f"Warning: {len(embeddings) - valid_count} invalid embeddings from {provider_name}")
                # Try to fix invalid embeddings
                for i, emb in enumerate(embeddings):
                    if not self.providers[provider_name].validate_embedding(emb):
                        # Recreate this single embedding
                        embeddings[i] = await self.create_embedding(texts[i], provider_name)
                return embeddings
                
        except Exception as e:
            latency = time.time() - start_time
            self._record_performance(provider_name, False, latency)
            print(f"Batch error with {provider_name} provider: {e}")
            
            # Try fallback providers
            for fallback in self.providers:
                if fallback != provider_name:
                    try:
                        print(f"Trying fallback provider for batch: {fallback}")
                        return await self.providers[fallback].create_embeddings_batch(texts)
                    except Exception as fallback_error:
                        print(f"Fallback {fallback} also failed: {fallback_error}")
            
            # All providers failed, create embeddings one by one
            print("All batch providers failed, creating embeddings individually")
            embeddings = []
            for text in texts:
                embeddings.append(await self.create_embedding(text))
            return embeddings
    
    def get_dimension(self, provider: Optional[str] = None) -> int:
        """Get the dimension of embeddings for the specified or current provider."""
        provider_name = provider or self.current_provider
        if provider_name in self.providers:
            return self.providers[provider_name].dimension
        return 768  # Default fallback
    
    def get_provider_info(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific provider or the current provider."""
        provider_name = provider or self.current_provider
        if provider_name in self.providers:
            return self.providers[provider_name].get_metadata()
        return {}
    
    def list_providers(self) -> List[str]:
        """List all available providers."""
        return list(self.providers.keys())
    
    def switch_provider(self, provider: str):
        """Switch to a different embedding provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not available. Available: {self.list_providers()}")
        self.current_provider = provider
        print(f"Switched to {provider} embedding provider")
    
    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all providers."""
        stats = {}
        
        for provider_name, history in self.performance_history.items():
            if history:
                recent = history[-100:]  # Last 100 operations
                success_count = sum(1 for h in recent if h["success"])
                total_count = len(recent)
                
                stats[provider_name] = {
                    "total_operations": len(history),
                    "recent_operations": total_count,
                    "success_rate": success_count / total_count if total_count > 0 else 0,
                    "avg_latency": sum(h["latency"] for h in recent if h["success"]) / max(1, success_count),
                    "total_tokens": sum(h.get("tokens", 0) for h in history),
                    "cost_per_token": self.providers[provider_name].get_cost_per_token() if provider_name in self.providers else 0
                }
        
        return stats
    
    async def create_multimodal_embedding(self, content: Dict[str, Any], provider: Optional[str] = None) -> List[float]:
        """
        Create an embedding for multimodal content (if supported by provider).
        
        Args:
            content: Dictionary with 'text' and/or 'image_path' keys
            provider: Specific provider to use (optional)
            
        Returns:
            List of floats representing the embedding
        """
        provider_name = provider or self.current_provider
        
        # Check if provider supports multimodal
        if provider_name == "cohere" and isinstance(self.providers.get("cohere"), CohereEmbeddingProvider):
            if self.providers["cohere"].use_v2:
                return await self.providers["cohere"].create_multimodal_embedding(content)
        
        # Fallback to text-only embedding
        if "text" in content:
            return await self.create_embedding(content["text"], provider_name)
        else:
            raise ValueError("No text content provided for embedding")