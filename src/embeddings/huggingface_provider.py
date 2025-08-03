"""
HuggingFace embedding provider implementation.
"""
import os
import time
from typing import List, Optional, Dict, Any
import numpy as np
from .base import BaseEmbeddingProvider

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# Try to import HuggingFace Inference API client
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface-hub not installed. Install with: pip install huggingface-hub")


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider with support for local and API inference."""
    
    def __init__(
        self, 
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_api: bool = False,
        api_token: Optional[str] = None,
        device: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize HuggingFace embedding provider.
        
        Args:
            model: HuggingFace model name or path
            use_api: Whether to use HuggingFace Inference API instead of local
            api_token: HuggingFace API token (required for API usage)
            device: Device to run the model on (cpu, cuda, mps, etc.)
            dimension: Embedding dimension (will be auto-detected if not provided)
        """
        # Common HuggingFace embedding models and their dimensions
        default_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L12-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/all-distilroberta-v1": 768,
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
            "sentence-transformers/multi-qa-mpnet-base-cos-v1": 768,
            "sentence-transformers/multi-qa-distilbert-cos-v1": 768,
            "sentence-transformers/msmarco-MiniLM-L-6-v3": 384,
            "sentence-transformers/msmarco-distilbert-base-v4": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "thenlper/gte-small": 384,
            "thenlper/gte-base": 768,
            "thenlper/gte-large": 1024,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024
        }
        
        # Use provided dimension or default
        if dimension is None:
            dimension = default_dimensions.get(model, 768)
        
        super().__init__(model, dimension)
        
        self.use_api = use_api
        self.device = device
        
        if use_api:
            # Initialize API client
            if not HF_HUB_AVAILABLE:
                raise ImportError("huggingface-hub is required for API usage. Install with: pip install huggingface-hub")
            
            self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
            if not self.api_token:
                raise ValueError("HuggingFace API token required for API usage. Set HUGGINGFACE_API_TOKEN environment variable.")
            
            self.client = InferenceClient(token=self.api_token)
            
            # Test API connection and get actual dimension
            self._test_api_connection()
        else:
            # Initialize local model
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers is required for local inference. Install with: pip install sentence-transformers")
            
            print(f"Loading model {model}...")
            self.model = SentenceTransformer(model, device=device)
            
            # Get actual dimension from model
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)
            print(f"Model loaded. Embedding dimension: {self.dimension}")
        
        # Set metadata
        self.metadata = {
            "supports_batch": True,
            "max_batch_size": 512 if not use_api else 96,
            "is_local": not use_api,
            "device": device or "cpu"
        }
    
    def _test_api_connection(self):
        """Test HuggingFace API connection and get embedding dimension."""
        try:
            # Create a test embedding
            response = self.client.feature_extraction(
                "test",
                model=self.model_name
            )
            
            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], list):
                    # Response is [embedding]
                    self.dimension = len(response[0])
                else:
                    # Response is embedding
                    self.dimension = len(response)
                print(f"API connection successful. Embedding dimension: {self.dimension}")
            else:
                raise ValueError("Unexpected response format from HuggingFace API")
                
        except Exception as e:
            raise ConnectionError(f"Failed to connect to HuggingFace API: {e}")
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            embeddings = await self.create_embeddings_batch([text])
            return embeddings[0] if embeddings else [0.0] * self.dimension
        except Exception as e:
            print(f"Error creating HuggingFace embedding: {e}")
            return [0.0] * self.dimension
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
        
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                if self.use_api:
                    # Use HuggingFace Inference API
                    embeddings = []
                    
                    # API typically handles one at a time or small batches
                    batch_size = 10
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        
                        # API call
                        response = self.client.feature_extraction(
                            batch,
                            model=self.model_name
                        )
                        
                        # Handle response format
                        if isinstance(response[0], list):
                            embeddings.extend(response)
                        else:
                            # Single text response
                            embeddings.append(response)
                    
                    return embeddings
                else:
                    # Use local model
                    embeddings = self.model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=len(texts) > 50,
                        batch_size=32
                    )
                    
                    # Convert numpy arrays to lists
                    return embeddings.tolist()
                    
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                    # Create embeddings individually as fallback
                    embeddings = []
                    successful_count = 0
                    
                    for i, text in enumerate(texts):
                        try:
                            if self.use_api:
                                response = self.client.feature_extraction(
                                    text,
                                    model=self.model_name
                                )
                                embedding = response[0] if isinstance(response[0], list) else response
                            else:
                                embedding = self.model.encode(text, convert_to_numpy=True).tolist()
                            
                            embeddings.append(embedding)
                            successful_count += 1
                        except Exception as individual_error:
                            print(f"Failed to create embedding for text {i}: {individual_error}")
                            embeddings.append([0.0] * self.dimension)
                    
                    print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                    return embeddings
    
    def get_cost_per_token(self) -> float:
        """
        Get the cost per token for HuggingFace embeddings.
        
        Returns:
            Cost per token (0 for local, minimal for API)
        """
        if self.use_api:
            # HuggingFace Inference API pricing varies by model
            # Using a rough estimate
            return 0.00001 / 1000  # Very low cost
        else:
            return 0.0  # Local inference has no API costs
    
    def encode_with_instruction(self, texts: List[str], instruction: str) -> List[List[float]]:
        """
        Encode texts with a specific instruction (for models that support it).
        
        Args:
            texts: List of texts to encode
            instruction: Instruction to prepend to texts
            
        Returns:
            List of embeddings
        """
        if not self.use_api and hasattr(self.model, 'encode'):
            # Some models like instructor-embeddings support instructions
            if hasattr(self.model, 'encode_with_prompt'):
                embeddings = self.model.encode_with_prompt(
                    [(instruction, text) for text in texts],
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            else:
                # Fallback: prepend instruction to texts
                instructed_texts = [f"{instruction}: {text}" for text in texts]
                return self.model.encode(instructed_texts, convert_to_numpy=True).tolist()
        else:
            # For API or models without instruction support
            instructed_texts = [f"{instruction}: {text}" for text in texts]
            return self.create_embeddings_batch(instructed_texts)
    
    async def encode_multi_modal(self, texts: List[str], images: Optional[List[Any]] = None) -> List[List[float]]:
        """
        Encode multi-modal inputs (for models that support it).
        
        Args:
            texts: List of texts
            images: List of images (PIL Images or paths)
            
        Returns:
            List of embeddings
        """
        if not self.use_api and hasattr(self.model, 'encode_multi_modal'):
            # Some models support multi-modal encoding
            embeddings = self.model.encode_multi_modal(
                texts=texts,
                images=images,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        else:
            # Fallback to text-only encoding
            print("Warning: Multi-modal encoding not supported by this model. Using text-only.")
            return await self.create_embeddings_batch(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "use_api": self.use_api
        }
        
        if not self.use_api and hasattr(self.model, 'get_config_dict'):
            config = self.model.get_config_dict()
            info.update({
                "max_seq_length": config.get("max_seq_length", "unknown"),
                "pooling_mode": config.get("pooling_mode", "unknown")
            })
        
        return info