import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import time
from tqdm import tqdm
import easyocr
from paddleocr import PaddleOCR

from app.core.logging import logger
from app.core.config import config

class DocumentIngestionService:
    """Handles document ingestion from PDF files (both searchable and scanned)."""
    
    def __init__(self):
        self.config = config.get_section("ingestion")
        self.ocr_backend = self.config.get("ocr_backend", "paddle")
        self.tesseract_lang = self.config.get("tesseract_lang", "ben")
        self.render_dpi = self.config.get("render_dpi", 300)
        self.progress_interval = self.config.get("progress_interval", 10)
        
        # Initialize OCR engines
        self.paddle_ocr = None
        self.easy_ocr = None
        
        if self.ocr_backend == "paddle":
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # multilingual support
                use_gpu=False,
                show_log=False
            )
        else:
            self.easy_ocr = easyocr.Reader([self.tesseract_lang], gpu=False)
    
    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main ingestion pipeline for PDF documents."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Starting ingestion of {pdf_path.name}", 
                   pdf_path=str(pdf_path))
        
        start_time = time.time()
        
        try:
            # Detect if PDF is searchable
            is_searchable = self._is_searchable_pdf(pdf_path)
            logger.info(f"PDF type detected: {'searchable' if is_searchable else 'scanned'}")
            
            if is_searchable:
                text, metadata = self._extract_text_searchable(pdf_path)
            else:
                text, metadata = self._extract_text_scanned(pdf_path)
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            ingestion_time = time.time() - start_time
            logger.info(f"Ingestion completed in {ingestion_time:.2f}s",
                       num_pages=metadata.get("num_pages"),
                       char_count=len(cleaned_text),
                       ingestion_time=ingestion_time)
            
            return {
                "text": cleaned_text,
                "metadata": {
                    **metadata,
                    "source": str(pdf_path),
                    "ingestion_time": ingestion_time,
                    "is_searchable": is_searchable
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF {pdf_path.name}", 
                        error=str(e), pdf_path=str(pdf_path))
            raise
    
    def _is_searchable_pdf(self, pdf_path: Path) -> bool:
        """Detect if PDF contains extractable text."""
        try:
            doc = fitz.open(pdf_path)
            total_chars = 0
            
            for page in doc:
                text = page.get_text()
                total_chars += len(text.strip())
                if total_chars > 100:  # Threshold for searchable content
                    break
            
            doc.close()
            return total_chars > 100
        except Exception:
            return False
    
    def _extract_text_searchable(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from searchable PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        full_text = []
        metadata = {
            "num_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", "")
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text.append(text)
            
            # Log progress
            if (page_num + 1) % self.progress_interval == 0:
                logger.debug(f"Extracted page {page_num + 1}/{len(doc)}")
        
        doc.close()
        return "\n".join(full_text), metadata
    
    def _extract_text_scanned(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from scanned PDF using OCR."""
        doc = fitz.open(pdf_path)
        full_text = []
        metadata = {
            "num_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", "")
        }
        
        # Progress bar for OCR
        pbar = tqdm(total=len(doc), desc="OCR Processing", unit="page")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=self.render_dpi)
            img_data = pix.tobytes("png")
            
            # Convert to numpy array for OCR
            img = Image.open(io.BytesIO(img_data))
            img_np = np.array(img)
            
            # Perform OCR
            if self.ocr_backend == "paddle":
                result = self.paddle_ocr.ocr(img_np, cls=True)
                text = self._extract_text_paddle(result)
            else:
                result = self.easy_ocr.readtext(img_np)
                text = self._extract_text_easyocr(result)
            
            full_text.append(text)
            pbar.update(1)
            
            # Log progress
            if (page_num + 1) % self.progress_interval == 0:
                logger.debug(f"OCR completed page {page_num + 1}/{len(doc)}")
        
        pbar.close()
        doc.close()
        return "\n".join(full_text), metadata
    
    def _extract_text_paddle(self, result: List) -> str:
        """Extract text from PaddleOCR result."""
        text_lines = []
        for line in result:
            if line:
                for word_info in line:
                    text_lines.append(word_info[1][0])
        return " ".join(text_lines)
    
    def _extract_text_easyocr(self, result: List) -> str:
        """Extract text from EasyOCR result."""
        text_lines = []
        for detection in result:
            text_lines.append(detection[1])
        return " ".join(text_lines)
    
    def _clean_text(self, text: str) -> str:
        """Apply basic cleaning to extracted text."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def perform_ocr_quality_check(self, text: str) -> Dict[str, Any]:
        """Check OCR quality metrics."""
        total_chars = len(text)
        if total_chars == 0:
            return {"score": 0, "status": "failed", "message": "No text extracted"}
        
        # Count Bengali characters
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        bengali_ratio = bengali_chars / total_chars if total_chars > 0 else 0
        
        # Check for common OCR artifacts
        has_artifacts = any(
            artifact in text.lower()
            for artifact in ['ocr', '©', '®', '™']
        )
        
        quality_score = min(1.0, bengali_ratio * 1.2)  # Penalize non-Bengali content
        
        return {
            "score": quality_score,
            "status": "acceptable" if quality_score > 0.3 else "poor",
            "bengali_ratio": bengali_ratio,
            "has_artifacts": has_artifacts,
            "char_count": total_chars
        }