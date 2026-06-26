from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional
import numpy as np
from app.core.logging import logger
from app.core.config import config

class RerankingService:
    """Reranking service using cross-encoder models."""
    
    def __init__(self):
        self.config = config.get_section("reranking")
        self.enabled = self.config.get("enabled", True)
        self.model_name = self.config.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.fallback_model = self.config.get("fallback_model", "BAAI/bge-reranker-base")
        self.top_k = self.config.get("top_k", 10)
        
        self.model = None
        if self.enabled:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the reranking model."""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranking model initialized: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}, using fallback")
            try:
                self.model = CrossEncoder(self.fallback_model)
                logger.info(f"Using fallback reranking model: {self.fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                self.enabled = False
                self.model = None
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query."""
        if not documents or not self.enabled or not self.model:
            return documents
        
        if top_k is None:
            top_k = self.top_k
        
        logger.debug(f"Reranking {len(documents)} documents")
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, doc['text']) for doc in documents]
            
            # Get scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Add scores to documents
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])
            
            # Sort by rerank score
            sorted_docs = sorted(documents, 
                               key=lambda x: x.get('rerank_score', 0), 
                               reverse=True)
            
            # Store original scores for reference
            for doc in sorted_docs:
                if 'combined_score' not in doc:
                    doc['combined_score'] = doc.get('rerank_score', 0)
                doc['original_score'] = doc.get('combined_score', 0)
                doc['combined_score'] = doc['rerank_score']  # Use rerank score as final
            
            return sorted_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
    def compute_rerank_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Compute reranking scores without modifying documents."""
        if not documents or not self.enabled or not self.model:
            return [1.0] * len(documents)
        
        try:
            pairs = [(query, doc['text']) for doc in documents]
            scores = self.model.predict(pairs)
            return scores.tolist()
        except Exception as e:
            logger.error(f"Failed to compute rerank scores: {e}")
            return [1.0] * len(documents)