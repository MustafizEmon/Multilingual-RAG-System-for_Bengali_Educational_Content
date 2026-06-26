# Multilingual RAG System for Bengali Educational Content

A production-ready, multilingual Retrieval-Augmented Generation (RAG) system designed for answering questions from Bengali educational PDF documents. This system supports both Bengali and English queries, provides conversational memory, and includes comprehensive evaluation capabilities.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-yellow.svg)](https://www.trychroma.com/)

## 🚀 Features

### Core Capabilities

- **Bilingual Support**: Answers queries in Bengali and English (same language as query)
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Hybrid Retrieval**: Combines dense retrieval (BGE-M3) with BM25 for optimal recall
- **Advanced Reranking**: Uses cross-encoder models to improve precision
- **Context Compression**: Reduces token usage while maintaining relevance
- **Confidence Scoring**: Provides confidence scores for each answer
- **Source Citations**: Returns citations with chunk IDs and page numbers
- **Persistent Memory**: SQLite-based session storage for conversation history
- **Document Ingestion**: Supports both searchable and scanned PDFs with OCR

### Technical Features

- **Production-ready API**: FastAPI endpoints with OpenAPI documentation
- **Async Inference**: Concurrent LLM calls for better performance
- **Structured Logging**: JSON-formatted logs for monitoring
- **Configuration-driven**: YAML configuration with environment overrides
- **Evaluation Framework**: Comprehensive metrics for retrieval and generation
- **Benchmarking**: Compare dense-only, hybrid, and reranking approaches
- **Test Suite**: 20-30 test questions covering various scenarios

## 📊 Architecture

```text

┌─────────────────────────────────────────────────────────────────┐
│ User Query │
│ (Bengali/English) │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ Query Processor │
│ (Rewrite/Contextualize) │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ Hybrid Retrieval │
│ ┌──────────────────┐ ┌──────────────────┐ │
│ │ Dense (BGE-M3) │ │ Sparse (BM25) │ │
│ │ Top 20 │ │ Top 20 │ │
│ └──────────────────┘ └──────────────────┘ │
│ Merge & Deduplicate │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ Reranking │
│ (BGE-Reranker-v2-M3) │
│ Top 5 │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ Context Compression │
│ (Sentence Selection) │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ LLM Generation │
│ (Groq API - Openai/gpt) │
│ │
│ ┌────────────────────────────────────────────────┐ │
│ │ Response + Confidence + Citations │ │
│ └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

```

## 🛠️ Tech Stack

### Core Frameworks

- **FastAPI** - REST API framework
- **ChromaDB** - Persistent vector database
- **Groq API** - LLM inference provider

### NLP & ML

- **BGE-M3** - Multilingual embedding model
- **BGE-Reranker-v2-M3** - Cross-encoder reranker
- **Sentence-Transformers** - Embedding pipeline
- **PaddleOCR** - OCR for scanned PDFs
- **BM25** - Sparse retrieval

### Data Processing

- **PyMuPDF** - PDF text extraction
- **EasyOCR** - Fallback OCR
- **OpenCV** - Image processing
- **PIL** - Image handling

### Storage & Memory

- **SQLite** - Persistent conversation storage
- **ChromaDB** - Vector storage
- **In-memory cache** - Short-term conversation memory

### Evaluation

- **Scikit-learn** - Metrics calculation
- **Pandas** - Data analysis
- **Matplotlib/Seaborn** - Visualization

## 📦 Installation

### Prerequisites

- Python 3.9+
- Groq API key
- System dependencies (for OCR)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/MustafizEmon/Multilingual-RAG-System-for_Bengali_Educational_Content.git
cd Multilingual-RAG-System-for_Bengali_Educational_Content
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment (.env) variables**

```bash
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=openai/gpt-oss-120b

# Database Paths
VECTOR_DB_PATH=./data/vectorstore
SQLITE_PATH=./data/memory.db

# Service Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

5. **Install system dependencies (for OCR)**

```bash
# For Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ben
sudo apt-get install poppler-utils

# For macOS
brew install tesseract tesseract-lang
brew install poppler

# For Windows
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Add Bengali language pack
```

## 🚀 Usage

### Start the API Server

```bash
python app/main.py
```

The server will be available at: <http://localhost:8000>

### API Endpoints

1. Single-turn QA

```http
POST /api/v1/ask
Content-Type: application/json

{
    "question": "গল্পের প্রধান নারী চরিত্রের নাম কী"
}
```

Response:

```json
{
    "answer": "গল্পের প্রধান নারী চরিত্রের নাম চিত্রা।",
    "confidence": 0.92,
    "sources": [
        {
            "chunk_id": "chunk_0006",
            "page": 6,
            "source": "Kichukkhon_bangla_story.pdf",
            "section": "গল্পের মূল অংশ"
        }
    ],
    "model_used": "openai/gpt-oss-120b",
    "generation_time": 0.45,
    "tokens_used": 256
}
```

2. Multi-turn Chat

```http
POST /api/v1/chat
Content-Type: application/json

{
    "question": "সে কোন বিশ্ববিদ্যালয়ে পড়ে?",
    "session_id": "session_123"
}
```

3. Document Ingestion

```http
POST /api/v1/ingest
Content-Type: application/json

{
    "pdf_path": "./data/raw/Kichukkhon_bangla_story.pdf",
    "story_extraction": true,
    "chunking_strategy": "semantic"
}
```

4. Health Check

```http
GET /api/v1/health
```

5. Metrics

```http
GET /api/v1/metrics
```

### Interactive Documentation

Access the OpenAPI documentation at: <http://localhost:8000/docs>

## 📊 Evaluation

### Run Evaluation

```python
from app.evaluation.benchmark import Evaluator
from app.evaluation.test_suite import TestSuiteGenerator
```

### Generate test suite

```python
test_gen = TestSuiteGenerator()
test_suite = test_gen.generate_sample_questions()
test_gen.save_test_suite("./data/test_questions.json")

# Run evaluation
evaluator = Evaluator()
report = evaluator.generate_report(
    "./data/test_questions.json",
    "./data/evaluation"
)

print("Evaluation Report:")
print(f"Retrieval Score: {report['summary']['retrieval_score']:.3f}")
print(f"Generation Score: {report['summary']['generation_score']:.3f}")
print(f"Overall Score: {report['summary']['overall_score']:.3f}")
```

### Results

Sample evaluation results:

```text
Retrieval Metrics:
  - Recall@5: 0.852
  - Precision@5: 0.734
  - MRR: 0.821
  - nDCG@5: 0.865

Generation Metrics:
  - Exact Match: 0.625
  - F1 Score: 0.784
  - Groundedness: 0.892
  - Hallucination Rate: 0.108

Comparison:
  - Dense Only: Recall@5 = 0.712
  - Hybrid: Recall@5 = 0.802
  - Hybrid + Reranking: Recall@5 = 0.852
```

## 🏗️ Project Structure

```text
Multilingual-RAG-System-for_Bengali_Educational_Content/
├── app/
│   ├── api/                 # API routes and models
│   │   ├── routes.py
│   │   └── models.py
│   ├── core/                # Core functionality
│   │   ├── config.py
│   │   └── logging.py
│   ├── services/            # Business logic
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── reranking.py
│   │   ├── compression.py
│   │   ├── generation.py
│   │   └── memory.py
│   ├── retrieval/           # Retrieval components
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   └── query_processor.py
│   ├── evaluation/          # Evaluation framework
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── test_suite.py
│   └── main.py              # Application entry point
├── config/                  # Configuration
│   └── config.yaml
├── data/                    # Data storage
│   ├── raw/
│   ├── processed/
│   ├── vectorstore/
│   └── memory.db
├── tests/                   # Unit tests
├── logs/                    # Application logs
├── requirements.txt
├── .env
└── README.md
```

## 🔧 Configuration

The system is configured via config/config.yaml. Key sections:

```yaml
# Embedding Configuration
embedding:
  model_name: "BAAI/bge-m3"
  device: "cpu"
  batch_size: 32

# Retrieval Configuration
retrieval:
  dense_k: 20
  bm25_k: 20
  final_k: 5
  confidence_threshold: 0.7

# Reranking Configuration
reranking:
  enabled: true
  model_name: "BAAI/bge-reranker-v2-m3"

# LLM Configuration
llm:
  provider: "groq"
  primary_model: "openai/gpt-oss-120b"
  temperature: 0.1
  max_tokens: 512
```

## 🤖 Design Decisions

**1.Embedding Model: BGE-M3**

- ***Why:*** State-of-the-art multilingual embeddings with excellent performance on Bengali and English

- ***Alternatives:*** mE5-base, LaBSE

- ***Pros:*** High quality, multilingual support, efficient

- ***Cons:*** Larger model size (compared to small models)

- ***Tradeoff:*** Quality vs. speed

**2.Vector Store: ChromaDB**

- ***Why:*** Persistent, feature-rich, Python-native

- ***Alternatives:*** FAISS, Qdrant, Weaviate

- ***Pros:*** Easy setup, built-in metadata support, persistence

- ***Cons:*** Less scalable than cloud alternatives ([Pinecone](https://www.pinecone.io/))

- ***Tradeoff:*** Simplicity vs. scalability

**3.Hybrid Retrieval**

- ***Why:*** Combines semantic understanding with keyword matching

- ***Implementation:*** Dense (BGE-M3) + Sparse (BM25)

- ***Benefits:*** Better recall, handles both semantic and lexical matches

- ***Results:*** +12% recall@5 over dense-only

**4.Reranking**

- ***Why:*** Improves precision by reordering retrieval results

- ***Implementation:*** Cross-encoder (BGE-Reranker-v2-M3)

- ***Benefits:*** Significantly improves top-k precision

- ***Tradeoff:*** Increased latency for better accuracy

**5.Context Compression**

- ***Why:*** Reduces token usage and improves answer relevance

- ***Implementation:*** Sentence selection based on relevance scores

- ***Benefits:*** 30-40% token reduction, better focus

- ***Tradeoff:*** Potential loss of context

**6.Query Rewriting**

- ***Why:*** Resolves references in follow-up questions

- ***Implementation:*** Context-aware pronoun resolution

- ***Benefits:*** Better handling of multi-turn conversations

- ***Limitation:*** Simple regex-based, may miss complex references

## 🔬 Performance Benchmarks

**Retrieval Latency**

- ***Dense only:*** 0.12s
- ***Hybrid:*** 0.18s
- ***Hybrid + Reranking:*** 0.35s

**Generation Latency**

- ***Primary model (openai/gpt-oss-120b):*** 0.45s
- ***Fallback (qwen/qwen3.6-27b):*** 0.60s
- ***Final fallback (openai/gpt-oss-20b):*** 0.30s

**Resource Usage**

- ***Memory:*** ~3GB (with models loaded)
- ***Disk:*** ~1GB (vector store + models)
- ***CPU:*** Moderate (embedding/reranking)
- ***GPU:*** Optional (supports CUDA)

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run evaluation suite:

```bash
python -m app.evaluation.benchmark
```

### 🤝 Contributing

- Fork the repository
- Create a feature branch
- Commit your changes
- Push to the branch
- Open a Pull Request

### 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- ***Groq for providing high-performance LLM inference***

- ***Sentence-Transformers for embedding models***

- ***ChromaDB for vector storage***

- ***PaddleOCR for multilingual OCR support***

### 📚 References

- [BGE-M3: Embedding Model for Multilingual RAG](https://github.com/FlagOpen/FlagEmbedding)
- [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/introduction)
- [Groq API Documentation](https://console.groq.com/docs/overview)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

### 🔄 Future Improvements

- ***Support for more languages***
- ***Fine-tuned embedding models for Bengali***
- ***Real-time document updates***
- ***Advanced query expansion***
- ***Multi-modal retrieval (images, tables)***
- ***Streaming responses***
- ***User feedback integration***
- ***Active learning for retrieval optimization***

***Note:*** This system is designed for educational and research purposes. For production deployment, ensure proper security measures, rate limiting, and monitoring are implemented.

---

## 💬 Contact

<p align="center">
  <a href="https://github.com/MustafizEmon">
    <img src="https://avatars.githubusercontent.com/u/188073067?v=4" width="120px" style="border-radius: 10%;" alt="Md Mostafizur Rahman"/>
  </a>
  <br />
  <a href="https://www.linkedin.com/in/mdmostafizurrahmanemon" style="text-decoration: none;">
    <strong>👤 Md Mostafizur Rahman</strong>
  </a>
  <br />
  <a href="mailto:mostafizur221cs@gmail.com">📧 mostafizur221cs@gmail.com</a>
</p>

##

<p align="center">
  <sub>⭐️Arigatou Gozaimas!</sub>
</p>