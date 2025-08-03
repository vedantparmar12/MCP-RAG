"""
Ollama embedding provider implementation for local embeddings.
"""
import requests
import os
import time
from typing import List, Optional, Dict, Any
from .base import BaseEmbeddingProvider


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local model inference."""
    
    def __init__(
        self, 
        model: str = "nomic-embed-text", 
        base_url: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model: Ollama model name to use for embeddings
            base_url: Ollama API base URL (defaults to http://localhost:11434)
            dimension: Embedding dimension (will be auto-detected if not provided)
        """
        # Common Ollama embedding models and their dimensions
        default_dimensions = {
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "mxbai-embed-large": 1024,
            "bge-small": 384,
            "bge-base": 768,
            "bge-large": 1024,
            "e5-small": 384,
            "e5-base": 768,
            "e5-large": 1024
        }
        
        # Use provided dimension or default
        if dimension is None:
            dimension = default_dimensions.get(model, 768)
        
        super().__init__(model, dimension)
        
        # Set base URL
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_endpoint = f"{self.base_url}/api/embeddings"
        
        # Test connection and auto-detect dimension if needed
        self._test_connection()
        
        # Set metadata
        self.metadata = {
            "supports_batch": False,  # Ollama processes one at a time
            "is_local": True,
            "base_url": self.base_url
        }
    
    def _test_connection(self):
        """Test connection to Ollama and auto-detect embedding dimension."""
        try:
            # Create a test embedding
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": "test"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    actual_dimension = len(data["embedding"])
                    if actual_dimension != self.dimension:
                        print(f"Auto-detected dimension {actual_dimension} for model {self.model_name}")
                        self.dimension = actual_dimension
            else:
                raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running (ollama serve) and the model is pulled."
            )
        except Exception as e:
            print(f"Warning: Could not test Ollama connection: {e}")
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text using Ollama's API.
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=60  # Longer timeout for local models
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])
                    
                    if self.validate_embedding(embedding):
                        return embedding
                    else:
                        print(f"Invalid embedding received from Ollama")
                        return [0.0] * self.dimension
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"Ollama request timed out (attempt {retry + 1}/{max_retries})")
            except Exception as e:
                print(f"Error creating Ollama embedding (attempt {retry + 1}/{max_retries}): {e}")
            
            if retry < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
        
        print(f"Failed to create embedding after {max_retries} attempts")
        return [0.0] * self.dimension
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        Note: Ollama doesn't support batch processing, so we process sequentially.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
        
        embeddings = []
        successful_count = 0
        
        print(f"Creating {len(texts)} embeddings with Ollama (sequential processing)...")
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                print(f"Progress: {i}/{len(texts)} embeddings created")
            
            embedding = await self.create_embedding(text)
            
            if self.validate_embedding(embedding):
                successful_count += 1
            
            embeddings.append(embedding)
        
        print(f"Successfully created {successful_count}/{len(texts)} embeddings")
        return embeddings
    
    def get_cost_per_token(self) -> float:
        """
        Get the cost per token for Ollama embeddings.
        
        Returns:
            0.0 since Ollama runs locally
        """
        return 0.0  # Local inference has no API costs
    
    async def pull_model(self, show_progress: bool = True) -> bool:
        """
        Pull the model from Ollama registry if not already available.
        
        Args:
            show_progress: Whether to show download progress
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if model exists
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Model {self.model_name} is already available")
                return True
            
            # Pull the model
            print(f"Pulling model {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=show_progress,
                timeout=None  # No timeout for downloads
            )
            
            if show_progress and response.headers.get('content-type') == 'application/x-ndjson':
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "pulling" in status or "downloading" in status:
                            print(f"\r{status}", end="", flush=True)
                print()  # New line after progress
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """
        List all available embedding models in Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                # Filter for embedding models (heuristic: models with 'embed' in name)
                embedding_models = [
                    model["name"] for model in models 
                    if "embed" in model["name"].lower() or 
                    any(tag in model["name"].lower() for tag in ["e5", "bge", "minilm"])
                ]
                
                return embedding_models
            
        except Exception as e:
            print(f"Error listing models: {e}")
        
        return []