"""
Cohere embedding provider implementation with multimodal support.
"""
import cohere
import base64
import os
import time
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .base import BaseEmbeddingProvider


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider with support for text and multimodal embeddings."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "embed-english-v3.0", use_v2: bool = False):
        """
        Initialize Cohere embedding provider.
        
        Args:
            api_key: Cohere API key (if not provided, will use COHERE_API_KEY env var)
            model: Cohere embedding model to use
            use_v2: Whether to use Cohere v2 API (required for multimodal embeddings)
        """
        # Model dimensions
        dimensions = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-v4.0": 1024  # v2 API model
        }
        
        super().__init__(model, dimensions.get(model, 1024))
        
        # Set API key
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not provided and COHERE_API_KEY environment variable not set")
        
        # Initialize client based on API version
        self.use_v2 = use_v2 or model == "embed-v4.0"
        if self.use_v2:
            self.client = cohere.ClientV2(api_key=self.api_key)
        else:
            self.client = cohere.Client(api_key=self.api_key)
        
        # Set metadata
        self.metadata = {
            "supports_batch": True,
            "max_batch_size": 96,
            "supports_multimodal": self.use_v2,
            "api_version": "v2" if self.use_v2 else "v1"
        }
    
    def _image_to_base64_data_url(self, image_path: str) -> str:
        """Convert an image file to base64 data URL format."""
        _, file_extension = os.path.splitext(image_path)
        file_type = file_extension[1:]  # Remove the dot
        
        with open(image_path, "rb") as f:
            enc_img = base64.b64encode(f.read()).decode("utf-8")
            enc_img = f"data:image/{file_type};base64,{enc_img}"
        return enc_img
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text using Cohere's API.
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        try:
            embeddings = await self.create_embeddings_batch([text])
            return embeddings[0] if embeddings else [0.0] * self.dimension
        except Exception as e:
            print(f"Error creating Cohere embedding: {e}")
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
                if self.use_v2:
                    # V2 API format
                    inputs = [{"content": [{"type": "text", "text": text}]} for text in texts]
                    response = self.client.embed(
                        model=self.model_name,
                        inputs=inputs,
                        input_type="search_document",
                        embedding_types=["float"]
                    )
                    return response.embeddings.float
                else:
                    # V1 API format
                    response = self.client.embed(
                        texts=texts,
                        model=self.model_name,
                        input_type="search_document",
                        truncate="END"
                    )
                    return response.embeddings
                    
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                    # Create embeddings individually as fallback
                    embeddings = []
                    successful_count = 0
                    
                    for i, text in enumerate(texts):
                        try:
                            if self.use_v2:
                                inputs = [{"content": [{"type": "text", "text": text}]}]
                                individual_response = self.client.embed(
                                    model=self.model_name,
                                    inputs=inputs,
                                    input_type="search_document",
                                    embedding_types=["float"]
                                )
                                embeddings.append(individual_response.embeddings.float[0])
                            else:
                                individual_response = self.client.embed(
                                    texts=[text],
                                    model=self.model_name,
                                    input_type="search_document",
                                    truncate="END"
                                )
                                embeddings.append(individual_response.embeddings[0])
                            successful_count += 1
                        except Exception as individual_error:
                            print(f"Failed to create embedding for text {i}: {individual_error}")
                            embeddings.append([0.0] * self.dimension)
                    
                    print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                    return embeddings
    
    async def create_multimodal_embedding(self, content: Dict[str, Any]) -> List[float]:
        """
        Create an embedding for multimodal content (text + image).
        
        Args:
            content: Dictionary with 'text' and/or 'image_path' keys
            
        Returns:
            List of floats representing the embedding
        """
        if not self.use_v2:
            raise ValueError("Multimodal embeddings require Cohere v2 API. Initialize with use_v2=True")
        
        content_list = []
        
        if "text" in content:
            content_list.append({"type": "text", "text": content["text"]})
        
        if "image_path" in content:
            base64_url = self._image_to_base64_data_url(content["image_path"])
            content_list.append({
                "type": "image_url", 
                "image_url": {"url": base64_url}
            })
        
        if not content_list:
            raise ValueError("Content must contain either 'text' or 'image_path'")
        
        try:
            doc_input = {"content": content_list}
            response = self.client.embed(
                model=self.model_name,
                inputs=[doc_input],
                input_type="search_document",
                embedding_types=["float"]
            )
            return response.embeddings.float[0]
        except Exception as e:
            print(f"Error creating multimodal embedding: {e}")
            return [0.0] * self.dimension
    
    async def create_query_embedding(self, query: str, input_type: str = "search_query") -> List[float]:
        """
        Create an embedding specifically for search queries.
        
        Args:
            query: Search query text
            input_type: Type of input ('search_query' or 'search_document')
            
        Returns:
            List of floats representing the embedding
        """
        try:
            if self.use_v2:
                inputs = [{"content": [{"type": "text", "text": query}]}]
                response = self.client.embed(
                    model=self.model_name,
                    inputs=inputs,
                    input_type=input_type,
                    embedding_types=["float"]
                )
                return response.embeddings.float[0]
            else:
                response = self.client.embed(
                    texts=[query],
                    model=self.model_name,
                    input_type=input_type,
                    truncate="END"
                )
                return response.embeddings[0]
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return [0.0] * self.dimension
    
    def get_cost_per_token(self) -> float:
        """
        Get the cost per token for Cohere embeddings.
        
        Returns:
            Cost per token in USD
        """
        # Costs as of 2024 (per 1000 embeddings, not per token)
        # Converting to approximate per-token cost
        costs = {
            "embed-english-v3.0": 0.00010 / 1000,
            "embed-multilingual-v3.0": 0.00010 / 1000,
            "embed-english-light-v3.0": 0.00002 / 1000,
            "embed-multilingual-light-v3.0": 0.00002 / 1000,
            "embed-v4.0": 0.00010 / 1000
        }
        return costs.get(self.model_name, 0.00010 / 1000)