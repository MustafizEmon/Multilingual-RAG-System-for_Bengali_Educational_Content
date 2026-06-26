import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timezone
from app.core.logging import logger
from app.core.config import config
from app.services.embedding import EmbeddingService

class VectorStore:
    """Vector database management with ChromaDB persistence."""
    
    def __init__(self):
        self.config = config.get_section("vectorstore")
        self.persist_dir = Path(self.config.get("persist_directory", "./data/vectorstore"))
        self.collection_name = self.config.get("collection_name", "bengali_documents")
        self.distance_metric = self.config.get("distance_metric", "cosine")
        
        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
        self.embedding_service = EmbeddingService()
        
        self._initialize_collection()
        
        logger.info(f"Vector store initialized at {self.persist_dir}", 
                   collection=self.collection_name)
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            # Check if collection exists
            existing_collections = self.client.list_collections()
            collection_names = [c.name for c in existing_collections]
            
            if self.collection_name in collection_names:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
                # Get collection metadata
                count = self.collection.count()
                logger.info(f"Collection contains {count} documents")
            else:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"created_at": datetime.now(timezone.utc).isoformat()}
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add document chunks to vector store."""
        if not chunks:
            return {"added": 0, "errors": 0}
        
        logger.info(f"Adding {len(chunks)} documents to vector store")
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_passages(chunks)
        
        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get("chunk_id", f"doc_{len(ids)}")
            ids.append(chunk_id)
            documents.append(chunk["text"])
            
            metadata = {
                "chunk_id": chunk_id,
                "page": chunk.get("page", 0),
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "timestamp": chunk.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1)
            }
            metadatas.append(metadata)
        
        try:
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(ids)} documents")
            return {"added": len(ids), "errors": 0}
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return {"added": 0, "errors": 1}
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                        "distance": results['distances'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "exists": self.collection is not None,
            "persist_dir": str(self.persist_dir)
        }
    
    def reset_collection(self):
        """Reset collection (delete all documents)."""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info(f"Collection {self.collection_name} reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False