import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import ndcg_score, precision_recall_fscore_support
from app.core.logging import logger
from app.services.embedding import EmbeddingService

class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def recall_at_k(self, relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_k.intersection(relevant_set)) / len(relevant_set)
    
    def precision_at_k(self, relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if not retrieved:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        return len(retrieved_k.intersection(relevant_set)) / k
    
    def mrr(self, relevant: List[str], retrieved: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        relevant_set = set(relevant)
        
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg(self, relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        if not relevant or not retrieved:
            return 0.0
        
        # Create relevance scores
        relevant_set = set(relevant)
        relevance_scores = [1.0 if doc in relevant_set else 0.0 for doc in retrieved[:k]]
        
        # Ideal scores
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        # Calculate DCG
        def dcg(scores):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))
        
        dcg_score = dcg(relevance_scores)
        ideal_dcg = dcg(ideal_scores)
        
        return dcg_score / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def cosine_similarity(self, query: str, doc: str) -> float:
        """Calculate cosine similarity between query and document."""
        query_emb = self.embedding_service.embed_query(query)
        doc_emb = self.embedding_service.embed_passages([{"text": doc}])[0]
        
        return np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))

class GenerationMetrics:
    """Generation evaluation metrics."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def exact_match(self, predicted: str, expected: str) -> bool:
        """Check exact match."""
        return predicted.strip() == expected.strip()
    
    def f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score for text."""
        # Tokenize
        pred_tokens = set(self._tokenize(predicted))
        exp_tokens = set(self._tokenize(expected))
        
        if not pred_tokens and not exp_tokens:
            return 1.0
        
        if not pred_tokens or not exp_tokens:
            return 0.0
        
        # Calculate precision and recall
        intersection = pred_tokens.intersection(exp_tokens)
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(exp_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def groundedness(self, answer: str, context: List[str]) -> float:
        """Check if answer is grounded in context."""
        # Check for factual claims
        answer_sentences = self._split_sentences(answer)
        context_text = " ".join(context)
        
        supported_claims = 0
        total_claims = len(answer_sentences)
        
        for sentence in answer_sentences:
            # Check if sentence is supported by context
            if self._is_supported(sentence, context_text):
                supported_claims += 1
        
        return supported_claims / total_claims if total_claims > 0 else 0.0
    
    def hallucination_rate(self, answer: str, context: List[str]) -> float:
        """Calculate hallucination rate."""
        return 1.0 - self.groundedness(answer, context)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        sentences = re.split(r'[.!?।]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_supported(self, sentence: str, context: str) -> bool:
        """Check if sentence is supported by context."""
        # Simple overlap check
        sentence_words = set(self._tokenize(sentence))
        context_words = set(self._tokenize(context))
        
        if not sentence_words:
            return True
        
        # Calculate overlap ratio
        overlap = len(sentence_words.intersection(context_words))
        return overlap / len(sentence_words) > 0.5