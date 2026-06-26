from typing import List, Dict, Any, Optional
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.logging import logger
from app.core.config import config

class ChunkingService:
    """Advanced text chunking with recursive and semantic strategies."""
    
    def __init__(self):
        self.config = config.get_section("chunking")
        self.strategy = self.config.get("strategy", "recursive")  #Recursive Chunking strategy were Used (config.yaml) 
        self.chunk_size = self.config.get("chunk_size", 800)
        self.chunk_overlap = self.config.get("chunk_overlap", 120)
        self.separators = self.config.get("separators", ["\n\n", "\n", "।", ".", "?", "!"])
        self.semantic_threshold = self.config.get("semantic_similarity_threshold", 0.7)  #if semantic is in use
        
        # Initialize embedding model for semantic chunking (if)
        self.embedding_model = None
        if self.strategy == "semantic":
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') #if semantic is in use but this model is not recommanded
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Main chunking pipeline."""
        if not text or not text.strip():
            return []
        
        
        logger.debug("Starting chunking", 
                    strategy=self.strategy,
                    text_length=len(text))
        
        if self.strategy == "recursive":
            chunks = self._recursive_chunking(text, metadata)
        else:
            chunks = self._semantic_chunking(text, metadata)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.update({
                "chunk_id": f"chunk_{i:04d}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": metadata.get("timestamp") if metadata else None,
                "source": metadata.get("source") if metadata else None,
                "page": metadata.get("page") if metadata else None,
                "section": metadata.get("section") if metadata else None,
            })
        
        # Log chunking statistics
        avg_chunk_size = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
        logger.info(f"Chunking complete: {len(chunks)} chunks, "
                    f"average size: {avg_chunk_size:.0f} characters")
        
        return chunks
    
    def _semantic_chunking(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Semantic chunking using sentence embeddings."""
        # Split into sentences using multiple separators
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [{"text": text}]
        
        # Get embeddings for sentences
        embeddings = self.embedding_model.encode(sentences)
        
        # Find semantic breakpoints
        breakpoints = [0]
        for i in range(1, len(sentences) - 1):
            similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if similarity < self.semantic_threshold:
                breakpoints.append(i)
        breakpoints.append(len(sentences))
        
        # Form chunks
        chunks = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1]
            chunk_text = " ".join(sentences[start:end])
            chunks.append({"text": chunk_text})
        
        return chunks
    
    def _recursive_chunking(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Recursive text chunking with overlap."""
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            # Find the end position for this chunk
            end_pos = min(current_pos + self.chunk_size, text_length)
            
            # Try to break at a separator
            chunk_end = end_pos
            for sep in self.separators:
                # Find the last occurrence of this separator within the chunk
                sep_pos = text.rfind(sep, current_pos, end_pos)
                if sep_pos != -1 and sep_pos > current_pos:
                    chunk_end = sep_pos + len(sep)
                    break
            
            # If no separator found, use the hard limit
            if chunk_end == end_pos:
                chunk_end = end_pos
            
            # Extract chunk
            chunk_text = text[current_pos:chunk_end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text})
            
            # Move position with overlap
            current_pos = chunk_end - self.chunk_overlap if chunk_end < text_length else text_length
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple separators."""
        # Create regex pattern from separators
        sep_pattern = '|'.join(re.escape(sep) for sep in self.separators)
        pattern = f'([^{sep_pattern}]+[{sep_pattern}]?)'
        
        sentences = re.findall(pattern, text)
        return [s.strip() for s in sentences if s.strip()]