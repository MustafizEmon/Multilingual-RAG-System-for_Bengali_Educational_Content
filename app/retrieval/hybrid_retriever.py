from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import re
from app.core.logging import logger
from app.core.config import config
from app.retrieval.vector_store import VectorStore

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse (BM25) retrieval."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.config = config.get_section("retrieval")
        self.dense_k = self.config.get("dense_k", 20)
        self.bm25_k = self.config.get("bm25_k", 20)
        self.final_k = self.config.get("final_k", 5)
        self.dense_weight = self.config.get("dense_weights", {}).get("dense", 0.6)
        self.bm25_weight = self.config.get("dense_weights", {}).get("bm25", 0.4)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        self.bm25_index = None
        self.documents = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from stored documents."""
        try:
            # Get all documents from vector store
            # Note: In production, you might want to maintain a separate index
            all_docs = self.vector_store.collection.get()
            
            if all_docs and 'documents' in all_docs:
                self.documents = all_docs['documents']
                
                # Tokenize documents for BM25
                tokenized_docs = [self._tokenize(doc) for doc in self.documents]
                self.bm25_index = BM25Okapi(tokenized_docs)
                
                logger.info(f"BM25 index built with {len(self.documents)} documents")
            else:
                logger.warning("No documents available for BM25 indexing")
                
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval."""
        if k is None:
            k = self.final_k
        
        logger.debug(f"Hybrid retrieval for query: {query[:100]}...")
        
        try:
            # Dense retrieval
            dense_results = self.vector_store.search(query, self.dense_k)
            dense_dict = {
                res['metadata']['chunk_id']: res 
                for res in dense_results
            } if dense_results else {}
            
            # Sparse (BM25) retrieval
            bm25_results = self._bm25_retrieve(query, self.bm25_k)
            bm25_dict = {
                res['chunk_id']: res 
                for res in bm25_results
            } if bm25_results else {}
            
            # Merge results
            merged_results = self._merge_results(dense_dict, bm25_dict)
            
            # Rerank merged results
            reranked_results = self._rerank_results(merged_results, query)
            
            # Apply confidence threshold
            final_results = self._apply_confidence_threshold(reranked_results)
            
            logger.debug(f"Retrieved {len(final_results)} results after filtering")
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # Fallback to dense retrieval only
            return self.vector_store.search(query, k)
    
    def _bm25_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve using BM25."""
        if not self.bm25_index or not self.documents:
            return []
        
        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top k documents
            top_indices = np.argsort(scores)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    # Try to get chunk_id from document (assuming it's in the text)
                    doc_text = self.documents[idx]
                    chunk_id = f"doc_{idx}"
                    
                    # Try to find chunk_id in metadata
                    try:
                        all_metadata = self.vector_store.collection.get(
                            ids=[chunk_id]
                        )
                        if all_metadata and all_metadata['metadatas']:
                            metadata = all_metadata['metadatas'][0]
                            chunk_id = metadata.get('chunk_id', chunk_id)
                    except:
                        pass
                    
                    results.append({
                        'chunk_id': chunk_id,
                        'text': doc_text,
                        'score': scores[idx],
                        'method': 'bm25'
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")
            return []
    
    def _merge_results(self, dense_dict: Dict, bm25_dict: Dict) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from dense and BM25 retrieval."""
        merged = {}
        
        # Add dense results
        for chunk_id, result in dense_dict.items():
            merged[chunk_id] = {
                'chunk_id': chunk_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'dense_score': result['score'],
                'bm25_score': 0.0,
                'combined_score': self.dense_weight * result['score']
            }
        
        # Add BM25 results
        for result in bm25_dict:
            chunk_id = result['chunk_id']
            if chunk_id in merged:
                merged[chunk_id]['bm25_score'] = result['score']
                merged[chunk_id]['combined_score'] = (
                    self.dense_weight * merged[chunk_id]['dense_score'] +
                    self.bm25_weight * result['score']
                )
            else:
                # Try to find in dense results by text
                text_match = None
                for dense_id, dense_result in dense_dict.items():
                    if dense_result['text'] == result['text']:
                        text_match = dense_id
                        break
                
                if text_match:
                    merged[text_match]['bm25_score'] = result['score']
                    merged[text_match]['combined_score'] = (
                        self.dense_weight * merged[text_match]['dense_score'] +
                        self.bm25_weight * result['score']
                    )
                else:
                    merged[chunk_id] = {
                        'chunk_id': chunk_id,
                        'text': result['text'],
                        'metadata': {},
                        'dense_score': 0.0,
                        'bm25_score': result['score'],
                        'combined_score': self.bm25_weight * result['score']
                    }
        
        # Convert to list and sort by combined score
        results = list(merged.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank results (placeholder - actual reranking done separately)."""
        # This is a simple reranking based on combined score
        # Actual reranking with cross-encoder will be done in RerankingService
        return results
    
    def _apply_confidence_threshold(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results below confidence threshold."""
        filtered = []
        for result in results:
            score = result.get('combined_score', 0)
            if score >= self.confidence_threshold:
                result['confidence'] = score
                filtered.append(result)
            else:
                logger.debug(f"Result below confidence threshold: {score:.3f}")
        
        return filtered
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents and update BM25 index."""
        # Add to vector store
        self.vector_store.add_documents(chunks)
        # Rebuild BM25 index
        self._build_bm25_index()