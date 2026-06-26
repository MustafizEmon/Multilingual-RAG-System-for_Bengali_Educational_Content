from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any
from datetime import datetime, timezone
import uuid
import asyncio

from app.api.models import (
    QueryRequest, ChatRequest, AnswerResponse, 
    HealthResponse, MetricsResponse,
    DocumentIngestRequest, DocumentIngestResponse,
    Citation
)
from app.core.config import config
from app.core.logging import logger
from app.services.ingestion import DocumentIngestionService
from app.services.preprocessing import TextPreprocessingService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_processor import QueryProcessor
from app.services.reranking import RerankingService
from app.services.compression import ContextCompressionService
from app.services.generation import GenerationService
from app.services.memory import ShortTermMemory, PersistentMemory

# Initialize services
router = APIRouter()

# Service instances (singleton pattern)
ingestion_service = DocumentIngestionService()
preprocessing_service = TextPreprocessingService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_store = VectorStore()
hybrid_retriever = HybridRetriever(vector_store)
short_term_memory = ShortTermMemory()
persistent_memory = PersistentMemory()
query_processor = QueryProcessor(short_term_memory)
reranking_service = RerankingService()
compression_service = ContextCompressionService(reranking_service)
generation_service = GenerationService()

# Metrics
query_counter = 0
query_latencies = []

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    """Single-turn QA endpoint."""
    global query_counter
    
    start_time = datetime.now(timezone.utc)
    query_counter += 1
    
    logger.info(f"Received question: {request.question[:100]}...")
    
    try:
        # Process query (no session for single-turn)
        processed_query = query_processor.process_query(
            request.question, 
            session_id=f"single_{uuid.uuid4().hex[:8]}"
        )
        
        # Detect language
        language = 'bn' if any('\u0980' <= c <= '\u09FF' for c in request.question) else 'en'
        
        # Retrieve relevant documents
        retrieved_docs = hybrid_retriever.retrieve(processed_query, k=20)
        
        # Check if we have enough relevant content
        if not retrieved_docs or retrieved_docs[0].get('combined_score', 0) < 0.3:
            return AnswerResponse(
                answer="উত্তর পাওয়া যায়নি" if language == 'bn' else "Answer not found",
                confidence=0.0,
                sources=[]
            )
        
        # Rerank
        reranked_docs = reranking_service.rerank(processed_query, retrieved_docs)
        
        # Compress context
        compressed_docs = compression_service.compress(processed_query, reranked_docs[:10])
        
        # Generate answer
        response = await generation_service.generate_response(
            processed_query, 
            compressed_docs,
            session_id="single",
            language=language
        )
        
        # Create citations
        sources = [
            Citation(
                chunk_id=cite.get('chunk_id', ''),
                page=cite.get('page', 0),
                source=cite.get('source', ''),
                section=cite.get('section', ''),
                text_preview=cite.get('text_preview', '')
            )
            for cite in response.get('citations', [])
        ]
        
        # Log latency
        end_time = datetime.now(timezone.utc)
        latency = (end_time - start_time).total_seconds()
        query_latencies.append(latency)
        
        return AnswerResponse(
            answer=response.get('answer', ''),
            confidence=response.get('confidence', 0.0),
            sources=sources,
            model_used=response.get('model', generation_service.primary_model),
            generation_time=response.get('generation_time', 0.0),
            tokens_used=response.get('tokens_used', 0)
        )
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=AnswerResponse)
async def chat(request: ChatRequest):
    """Multi-turn conversation endpoint."""
    global query_counter
    
    start_time = datetime.now(timezone.utc)
    query_counter += 1
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"Chat query for session {session_id}: {request.question[:100]}...")
    
    try:
        # Process query with context
        processed_query = query_processor.process_query(
            request.question, 
            session_id
        )
        
        # Detect language
        language = 'bn' if any('\u0980' <= c <= '\u09FF' for c in request.question) else 'en'
        
        # Retrieve relevant documents
        retrieved_docs = hybrid_retriever.retrieve(processed_query, k=20)
        
        # Check if we have enough relevant content
        if not retrieved_docs or retrieved_docs[0].get('combined_score', 0) < 0.3:
            return AnswerResponse(
                answer="উত্তর পাওয়া যায়নি" if language == 'bn' else "Answer not found",
                confidence=0.0,
                sources=[],
                session_id=session_id
            )
        
        # Rerank
        reranked_docs = reranking_service.rerank(processed_query, retrieved_docs)
        
        # Compress context
        compressed_docs = compression_service.compress(processed_query, reranked_docs[:10])
        
        # Generate answer
        response = await generation_service.generate_response(
            processed_query, 
            compressed_docs,
            session_id=session_id,
            language=language
        )
        
        # Store in memory
        short_term_memory.add_turn(
            session_id, 
            request.question, 
            response.get('answer', '')
        )
        persistent_memory.save_turn(
            session_id, 
            request.question, 
            response.get('answer', '')
        )
        
        # Create citations
        sources = [
            Citation(
                chunk_id=cite.get('chunk_id', ''),
                page=cite.get('page', 0),
                source=cite.get('source', ''),
                section=cite.get('section', ''),
                text_preview=cite.get('text_preview', '')
            )
            for cite in response.get('citations', [])
        ]
        
        # Log latency
        end_time = datetime.now(timezone.utc)
        latency = (end_time - start_time).total_seconds()
        query_latencies.append(latency)
        
        return AnswerResponse(
            answer=response.get('answer', ''),
            confidence=response.get('confidence', 0.0),
            sources=sources,
            session_id=session_id,
            model_used=response.get('model', generation_service.primary_model),
            generation_time=response.get('generation_time', 0.0),
            tokens_used=response.get('tokens_used', 0)
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_document(request: DocumentIngestRequest, 
                         background_tasks: BackgroundTasks):
    """Ingest a document into the system."""
    logger.info(f"Starting document ingestion: {request.pdf_path}")
    
    try:
        # Ingest document
        doc_result = ingestion_service.ingest_pdf(request.pdf_path)
        
        # Preprocess text
        cleaned_text = preprocessing_service.preprocess_text(doc_result['text'])
        
        # Extract story if requested
        if request.story_extraction:
            from app.services.story_extraction import StoryExtractionService
            story_service = StoryExtractionService()
            story_result = story_service.extract_story(cleaned_text)
            text_to_chunk = story_result['story']
        else:
            text_to_chunk = cleaned_text
        
        # Chunk text
        metadata = doc_result['metadata']
        chunks = chunking_service.chunk_text(text_to_chunk, metadata)
        
        # Add to vector store
        if chunks:
            result = vector_store.add_documents(chunks)
            
            # Update hybrid retriever
            hybrid_retriever.add_documents(chunks)
            
            return DocumentIngestResponse(
                status="success",
                documents_added=result.get('added', 0),
                chunks_created=len(chunks),
                metadata={
                    'source': request.pdf_path,
                    'total_chunks': len(chunks),
                    'method': story_result.get('method', 'none'),
                    'confidence': story_result.get('confidence', 0.0)
                }
            )
        else:
            return DocumentIngestResponse(
                status="failed",
                documents_added=0,
                chunks_created=0,
                metadata={'error': 'No chunks created'}
            )
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check vector store
    vector_stats = vector_store.get_collection_stats()
    vector_status = "healthy" if vector_stats.get('count', 0) > 0 else "empty"
    
    # Check LLM
    llm_status = "healthy" if generation_service.client else "unavailable"
    
    # Check memory
    memory_status = "healthy"
    
    return HealthResponse(
        status="healthy",
        version=config.get("app.version", "1.0.0"),
        timestamp=datetime.now(timezone.utc),
        vector_store_status=vector_status,
        llm_status=llm_status,
        memory_status=memory_status
    )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    # Calculate average latency
    avg_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0
    
    # Get vector store stats
    vector_stats = vector_store.get_collection_stats()
    
    # Get memory stats
    memory_stats = {
        'active_sessions': len(short_term_memory.memory),
        'total_stored_turns': sum(len(turns) for turns in short_term_memory.memory.values())
    }
    
    return MetricsResponse(
        total_queries=query_counter,
        average_latency=avg_latency,
        retrieval_stats={
            'collection_size': vector_stats.get('count', 0),
            'collection_name': vector_stats.get('name', ''),
            'last_retrieval_time': None
        },
        memory_usage=memory_stats,
        system_stats={
            'api_version': config.get("app.version", "1.0.0"),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    )