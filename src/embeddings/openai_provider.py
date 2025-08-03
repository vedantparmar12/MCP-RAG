"""
OpenAI embedding provider implementation.
"""
import openai
import os
import time
from typing import List, Optional
from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small model."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            model: OpenAI embedding model to use
        """
        # Model dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        super().__init__(model, dimensions.get(model, 1536))
        
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Set metadata
        self.metadata = {
            "supports_batch": True,
            "max_batch_size": 2048,
            "max_tokens": 8191
        }
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text using OpenAI's API.
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            embeddings = await self.create_embeddings_batch([text])
            return embeddings[0] if embeddings else [0.0] * self.dimension
        except Exception as e:
            print(f"Error creating OpenAI embedding: {e}")
            return [0.0] * self.dimension
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in a single API call.
        
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
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                    # Try creating embeddings one by one as fallback
                    print("Attempting to create embeddings individually...")
                    embeddings = []
                    successful_count = 0
                    
                    for i, text in enumerate(texts):
                        try:
                            individual_response = self.client.embeddings.create(
                                model=self.model_name,
                                input=[text]
                            )
                            embeddings.append(individual_response.data[0].embedding)
                            successful_count += 1
                        except Exception as individual_error:
                            print(f"Failed to create embedding for text {i}: {individual_error}")
                            # Add zero embedding as fallback
                            embeddings.append([0.0] * self.dimension)
                    
                    print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                    return embeddings
    
    def get_cost_per_token(self) -> float:
        """
        Get the cost per token for OpenAI embeddings.
        
        Returns:
            Cost per token in USD
        """
        # Costs as of 2024
        costs = {
            "text-embedding-3-small": 0.00002 / 1000,  # $0.020 per 1M tokens
            "text-embedding-3-large": 0.00013 / 1000,  # $0.130 per 1M tokens
            "text-embedding-ada-002": 0.00010 / 1000   # $0.100 per 1M tokens
        }
        return costs.get(self.model_name, 0.00002 / 1000)