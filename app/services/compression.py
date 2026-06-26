from typing import List, Dict, Any, Optional
import re
import numpy as np
from app.core.logging import logger
from app.core.config import config
from app.services.reranking import RerankingService

class ContextCompressionService:
    """Context compression to reduce token usage and improve relevance."""
    
    def __init__(self, reranking_service: RerankingService):
        self.config = config.get_section("compression")
        self.enabled = self.config.get("enabled", True)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.method = self.config.get("method", "sentence_compression")
        self.relevance_threshold = self.config.get("relevance_threshold", 0.6)
        
        self.reranking_service = reranking_service
    
    def compress(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress documents to most relevant content."""
        if not self.enabled or not documents:
            return documents
        
        logger.debug(f"Compressing {len(documents)} documents")
        
        # Estimate token count (rough approximation: 4 chars per token)
        total_chars = sum(len(doc['text']) for doc in documents)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= self.max_tokens:
            return documents
        
        if self.method == "sentence_compression":
            return self._compress_by_sentences(query, documents)
        elif self.method == "extractive_summary":
            return self._extractive_summary(query, documents)
        else:
            return self._truncate_by_relevance(documents)
    
    def _compress_by_sentences(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress by extracting most relevant sentences."""
        compressed_docs = []
        
        for doc in documents:
            # Split into sentences
            sentences = self._split_sentences(doc['text'])
            
            if len(sentences) <= 1:
                compressed_docs.append(doc)
                continue
            
            # Get rerank scores for each sentence
            sentence_docs = [{'text': s} for s in sentences]
            scores = self.reranking_service.compute_rerank_scores(query, sentence_docs)
            
            # Select sentences above threshold
            selected_sentences = []
            for sentence, score in zip(sentences, scores):
                if score >= self.relevance_threshold:
                    selected_sentences.append(sentence)
            
            # If no sentences selected, keep the most relevant one
            if not selected_sentences and scores:
                best_idx = np.argmax(scores)
                selected_sentences = [sentences[best_idx]]
            
            # Reconstruct compressed document
            if selected_sentences:
                compressed_text = ' '.join(selected_sentences)
                compressed_doc = doc.copy()
                compressed_doc['text'] = compressed_text
                compressed_doc['compressed'] = True
                compressed_doc['original_length'] = len(doc['text'])
                compressed_doc['compressed_length'] = len(compressed_text)
                compressed_docs.append(compressed_doc)
            else:
                compressed_docs.append(doc)
        
        return compressed_docs
    
    def _extractive_summary(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate extractive summary using textrank-like approach."""
        # Simplified implementation
        compressed_docs = []
        
        for doc in documents:
            sentences = self._split_sentences(doc['text'])
            
            if len(sentences) <= 3:
                compressed_docs.append(doc)
                continue
            
            # Simple sentence scoring: frequency + position
            word_freq = self._get_word_frequencies(doc['text'])
            
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                # Position score (first and last sentences get higher weight)
                position_weight = 1.0
                if i < 2:  # First two sentences
                    position_weight = 1.5
                elif i > len(sentences) - 2:  # Last two sentences
                    position_weight = 1.3
                
                # Word frequency score
                words = self._tokenize(sentence)
                freq_score = sum(word_freq.get(w, 0) for w in words)
                if words:
                    freq_score /= len(words)
                
                # Query relevance
                relevance = 0
                query_words = set(self._tokenize(query))
                if query_words:
                    relevance = sum(1 for w in words if w in query_words) / len(query_words)
                
                score = (freq_score * 0.4 + relevance * 0.6) * position_weight
                sentence_scores.append(score)
            
            # Select top sentences (up to 40% of original)
            num_selected = max(1, len(sentences) // 3)
            top_indices = np.argsort(sentence_scores)[-num_selected:][::-1]
            selected_sentences = [sentences[i] for i in sorted(top_indices)]
            
            compressed_text = ' '.join(selected_sentences)
            compressed_doc = doc.copy()
            compressed_doc['text'] = compressed_text
            compressed_doc['compressed'] = True
            
            compressed_docs.append(compressed_doc)
        
        return compressed_docs
    
    def _truncate_by_relevance(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate documents based on relevance scores."""
        # Sort by relevance score
        sorted_docs = sorted(documents, 
                           key=lambda x: x.get('combined_score', 0), 
                           reverse=True)
        
        # Keep documents until token limit reached
        compressed_docs = []
        current_tokens = 0
        
        for doc in sorted_docs:
            doc_tokens = len(doc['text']) // 4
            if current_tokens + doc_tokens <= self.max_tokens:
                compressed_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # Truncate the last document to fit
                remaining_tokens = self.max_tokens - current_tokens
                if remaining_tokens > 50:  # Only include if at least 50 tokens
                    truncated_text = ' '.join(doc['text'].split()[:remaining_tokens * 4])
                    truncated_doc = doc.copy()
                    truncated_doc['text'] = truncated_text
                    truncated_doc['truncated'] = True
                    compressed_docs.append(truncated_doc)
                break
        
        return compressed_docs
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting for Bengali/English
        text = re.sub(r'([.!?।؟])', r'\1|', text)
        sentences = [s.strip() for s in text.split('|') if s.strip()]
        return sentences
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    def _get_word_frequencies(self, text: str) -> Dict[str, float]:
        """Get word frequencies for text."""
        words = self._tokenize(text)
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        # Normalize
        max_freq = max(freq.values()) if freq else 1
        return {w: f / max_freq for w, f in freq.items()}