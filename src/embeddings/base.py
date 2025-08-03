"""
Base class for embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model_name: str, dimension: int):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Name of the embedding model
            dimension: Dimension of the embedding vectors
        """
        self.model_name = model_name
        self.dimension = dimension
        self.metadata = {}
    
    @abstractmethod
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        pass
    
    @abstractmethod
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in a single API call.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        """
        Get the cost per token for this embedding provider.
        
        Returns:
            Cost per token in USD
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the embedding provider.
        
        Returns:
            Dictionary containing provider metadata
        """
        return {
            "provider": self.__class__.__name__,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "cost_per_token": self.get_cost_per_token(),
            **self.metadata
        }
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the correct dimension and is not all zeros.
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if not embedding or len(embedding) != self.dimension:
            return False
        
        # Check if embedding is all zeros
        if all(v == 0.0 for v in embedding):
            return False
        
        return True