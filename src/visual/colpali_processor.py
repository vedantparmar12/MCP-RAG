"""
ColPali visual document processor for handling PDFs and visual documents.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Try to import ColPali dependencies
try:
    from colpali_engine.models import ColPali
    from transformers import AutoProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("Warning: ColPali not installed. Install with: pip install colpali-engine")

# Import Cohere for hybrid search
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


logger = logging.getLogger(__name__)


class ColPaliProcessor:
    """Processor for visual documents using ColPali."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize ColPali processor.
        
        Args:
            device: Device to run on (cuda, cpu)
        """
        if not COLPALI_AVAILABLE:
            raise ImportError("ColPali is not installed. Install with: pip install colpali-engine")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ColPali model
        print(f"Loading ColPali model on {self.device}...")
        self.model = ColPali.from_pretrained("vidore/colpali").to(self.device)
        self.processor = AutoProcessor.from_pretrained("vidore/colpali")
        self.model.eval()
        print("ColPali model loaded successfully")
    
    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image/PDF page.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process with ColPali
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
            
            return {
                "embeddings": embeddings.cpu(),
                "image_size": image.size,
                "image_path": image_path
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    async def process_batch(self, image_paths: List[str], batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        Process multiple images in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load images
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    batch_images.append(None)
            
            # Filter out failed loads
            valid_images = [img for img in batch_images if img is not None]
            valid_paths = [path for path, img in zip(batch_paths, batch_images) if img is not None]
            
            if valid_images:
                # Process batch
                inputs = self.processor(images=valid_images, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state
                
                # Create results
                for j, (path, emb) in enumerate(zip(valid_paths, embeddings)):
                    results.append({
                        "embeddings": emb.cpu(),
                        "image_size": valid_images[j].size,
                        "image_path": path
                    })
        
        return results
    
    def compute_late_interaction_score(
        self, 
        query_embeddings: torch.Tensor, 
        doc_embeddings: torch.Tensor
    ) -> float:
        """
        Compute ColBERT-style late interaction score.
        
        Args:
            query_embeddings: Query embeddings from ColPali
            doc_embeddings: Document embeddings from ColPali
            
        Returns:
            Similarity score
        """
        # Compute similarity matrix
        scores = torch.matmul(query_embeddings, doc_embeddings.transpose(-1, -2))
        
        # Max pooling over document dimension
        max_scores = scores.max(dim=-1).values
        
        # Sum over query tokens
        total_score = max_scores.sum().item()
        
        return total_score


class HybridDocumentRAG:
    """Hybrid RAG system combining ColPali for visual docs and Cohere for text/reranking."""
    
    def __init__(self, cohere_api_key: Optional[str] = None):
        """
        Initialize hybrid RAG system.
        
        Args:
            cohere_api_key: Cohere API key for reranking
        """
        # Initialize ColPali
        self.colpali_processor = ColPaliProcessor()
        
        # Initialize Cohere if available
        self.cohere_client = None
        self.cohere_v2_client = None
        
        if COHERE_AVAILABLE and cohere_api_key:
            self.cohere_client = cohere.Client(api_key=cohere_api_key)
            self.cohere_v2_client = cohere.ClientV2(api_key=cohere_api_key)
        
        # Storage
        self.visual_docs = []  # ColPali embeddings
        self.text_docs = []    # Text documents with embeddings
    
    async def add_visual_document(
        self, 
        doc_id: str, 
        image_path: str, 
        text_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a visual document (PDF page, screenshot, etc.).
        
        Args:
            doc_id: Document ID
            image_path: Path to the image
            text_content: Optional extracted text for reranking
            metadata: Optional metadata
        """
        # Process with ColPali
        result = await self.colpali_processor.process_image(image_path)
        
        if result:
            self.visual_docs.append({
                "id": doc_id,
                "colpali_embeddings": result["embeddings"],
                "text_content": text_content,
                "image_path": image_path,
                "metadata": metadata or {}
            })
    
    async def add_visual_documents_batch(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 4
    ):
        """
        Add multiple visual documents in batches.
        
        Args:
            documents: List of document dictionaries
            batch_size: Batch size for processing
        """
        # Extract image paths
        image_paths = [doc["image_path"] for doc in documents]
        
        # Process in batches
        results = await self.colpali_processor.process_batch(image_paths, batch_size)
        
        # Store results
        for doc, result in zip(documents, results):
            if result:
                self.visual_docs.append({
                    "id": doc["id"],
                    "colpali_embeddings": result["embeddings"],
                    "text_content": doc.get("text_content"),
                    "image_path": doc["image_path"],
                    "metadata": doc.get("metadata", {})
                })
    
    async def search_visual_documents(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search visual documents using ColPali.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.visual_docs:
            return []
        
        # Encode query with ColPali
        query_inputs = self.colpali_processor.processor(
            text=query, 
            return_tensors="pt"
        ).to(self.colpali_processor.device)
        
        with torch.no_grad():
            query_outputs = self.colpali_processor.model(**query_inputs)
            query_embeddings = query_outputs.last_hidden_state.cpu()
        
        # Compute scores
        scores = []
        for doc in self.visual_docs:
            score = self.colpali_processor.compute_late_interaction_score(
                query_embeddings, 
                doc["colpali_embeddings"]
            )
            scores.append((doc, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for doc, score in scores[:top_k]:
            results.append({
                "id": doc["id"],
                "score": score,
                "image_path": doc["image_path"],
                "text_content": doc.get("text_content"),
                "metadata": doc.get("metadata", {}),
                "type": "visual"
            })
        
        return results
    
    async def hybrid_search(
        self, 
        query: str, 
        visual_top_k: int = 10,
        text_top_k: int = 10,
        rerank_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across visual and text documents.
        
        Args:
            query: Search query
            visual_top_k: Top-k for visual search
            text_top_k: Top-k for text search
            rerank_top_k: Final top-k after reranking
            
        Returns:
            List of reranked results
        """
        results = []
        
        # Search visual documents
        if self.visual_docs:
            visual_results = await self.search_visual_documents(query, visual_top_k)
            results.extend(visual_results)
        
        # If we have Cohere and text documents, search those too
        if self.cohere_v2_client and self.text_docs:
            # This would integrate with the existing text search
            pass
        
        # Rerank if Cohere is available
        if self.cohere_client and results and any(r.get("text_content") for r in results):
            reranked = await self._rerank_results(query, results, rerank_top_k)
            return reranked
        
        return results[:rerank_top_k]
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Rerank results using Cohere."""
        # Prepare documents for reranking
        documents = []
        valid_indices = []
        
        for i, result in enumerate(results):
            if result.get("text_content"):
                documents.append(result["text_content"])
                valid_indices.append(i)
        
        if not documents:
            return results[:top_n]
        
        try:
            # Use Cohere reranker
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=documents,
                model="rerank-v3.5",
                top_n=min(top_n, len(documents))
            )
            
            # Map back to original results
            reranked_results = []
            for item in rerank_response.results:
                original_idx = valid_indices[item.index]
                result = results[original_idx].copy()
                result["rerank_score"] = item.relevance_score
                reranked_results.append(result)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results[:top_n]