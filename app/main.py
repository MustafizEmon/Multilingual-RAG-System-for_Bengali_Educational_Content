from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.config import config
from app.core.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG System", 
                version=config.get("app.version", "1.0.0"),
                debug=config.get("app.debug", False))
    yield
    # Shutdown
    logger.info("Shutting down RAG System")

# Create FastAPI app
app = FastAPI(
    title="Multilingual RAG System",
    description="Production-ready RAG system for Bengali/English educational content",
    version=config.get("app.version", "1.0.0"),
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "Multilingual RAG System",
        "version": config.get("app.version", "1.0.0"),
        "status": "running",
        "endpoints": {
            "/api/v1/ask": "Single-turn QA",
            "/api/v1/chat": "Multi-turn conversation",
            "/api/v1/ingest": "Document ingestion",
            "/api/v1/health": "Health check",
            "/api/v1/metrics": "Metrics"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8000),
        reload=config.get("app.debug", False)
    )