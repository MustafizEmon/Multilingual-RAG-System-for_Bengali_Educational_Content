from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from app.core.logging import logger
from app.core.config import config

class EmbeddingService:
    """Embedding pipeline for multilingual text."""
    
    def __init__(self):
        self.config = config.get_section("embedding")
        self.model_name = self.config.get("model_name", "BAAI/bge-m3")
        self.fallback_model = self.config.get("fallback_model", "intfloat/multilingual-e5-base")
        self.device = self.config.get("device", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        self.max_length = self.config.get("max_length", 512)
        self.query_prefix = self.config.get("query_prefix", "query: ")
        self.passage_prefix = self.config.get("passage_prefix", "passage: ")
        
        self.model = None
        self._initialize_model()
        
        logger.info(f"Embedding model initialized: {self.model_name}", 
                   device=self.device,
                   batch_size=self.batch_size)
    
    def _initialize_model(self):
        """Initialize the embedding model with fallback."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Test the model
            test_text = "Test embedding"
            self.model.encode([test_text], show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}, using fallback")
            try:
                self.model = SentenceTransformer(self.fallback_model, device=self.device)
                logger.info(f"Using fallback model: {self.fallback_model}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError("No embedding model available")
    
    def embed_passages(self, passages: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for passage chunks."""
        if not passages:
            return []
        
        texts = [p["text"] for p in passages]
        metadata = [p.get("metadata", {}) for p in passages]
        
        # Add passage prefix for BGE-M3
        prefixed_texts = [f"{self.passage_prefix}{t}" for t in texts]
        
        embeddings = []
        for i in range(0, len(prefixed_texts), self.batch_size):
            batch = prefixed_texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Try with smaller batch
                for text in batch:
                    emb = self.model.encode(
                        [text],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    embeddings.append(emb[0])
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query text."""
        prefixed_query = f"{self.query_prefix}{query}"
        
        try:
            embedding = self.model.encode(
                [prefixed_query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embedding[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          passage_embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute cosine similarity between query and passages."""
        if not passage_embeddings:
            return np.array([])
        
        passage_matrix = np.vstack(passage_embeddings)
        query_vector = query_embedding.reshape(1, -1)
        
        similarities = np.dot(query_vector, passage_matrix.T).flatten()
        return similarities