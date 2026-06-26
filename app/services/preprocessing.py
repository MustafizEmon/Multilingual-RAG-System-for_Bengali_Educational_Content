import re
import unicodedata
from typing import str, List, Dict, Any
from app.core.logging import logger
from app.core.config import config

class TextPreprocessingService:
    """Comprehensive text preprocessing for Bengali/English documents."""
    
    def __init__(self):
        self.config = config.get_section("preprocessing")
        
        # Bengali punctuation and special characters
        self.bengali_punct = '।॥৷'
        self.english_punct = '.!?'
        
        # Common OCR errors and corrections for Bengali
        self.ocr_corrections = {
            'ৰ': 'র',  # Assamese 'ro' to Bengali 'ro'
            'ৱ': 'ও',  # Assamese 'vo' to Bengali 'o'
            'ଓ': 'ও',  # Odia 'o' to Bengali 'o'
            'ସ': 'স',  # Odia 'so' to Bengali 'so'
            # Add more as needed
        }
    
    def preprocess_text(self, text: str) -> str:
        """Main preprocessing pipeline."""
        if not text:
            return ""
        
        logger.debug("Starting text preprocessing", original_length=len(text))
        
        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 2: Bengali normalization
        text = self._normalize_bengali(text)
        
        # Step 3: OCR correction
        text = self._correct_ocr_errors(text)
        
        # Step 4: Remove unwanted elements
        if self.config.get("remove_headers", True):
            text = self._remove_headers(text)
        
        if self.config.get("remove_footers", True):
            text = self._remove_footers(text)
        
        if self.config.get("remove_page_numbers", True):
            text = self._remove_page_numbers(text)
        
        # Step 5: Whitespace cleanup
        text = self._clean_whitespace(text)
        
        # Step 6: Remove duplicate lines
        text = self._remove_duplicate_lines(text)
        
        # Step 7: Newline normalization
        text = self._normalize_newlines(text)
        
        # Step 8: Preserve paragraphs
        if self.config.get("merge_paragraphs", True):
            text = self._preserve_paragraphs(text)
        
        logger.debug("Preprocessing complete", 
                    original_length=len(text), 
                    processed_length=len(text))
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Apply Unicode normalization."""
        return unicodedata.normalize(
            self.config.get("unicode_normalization", "NFKC"), 
            text
        )
    
    def _normalize_bengali(self, text: str) -> str:
        """Normalize Bengali-specific characters."""
        # Map common variations
        text = re.sub(r'\u09CE', '\u09CD\u09AF', text)  # Khanda ta
        text = re.sub(r'\u09F7', '\u09CD\u09B0', text)  # Raphala
        text = re.sub(r'\u09FA', '\u09CD\u09B2', text)  # Laphala
        
        # Normalize punctuation
        text = re.sub(r'[।॥]', '।', text)  # Single danda
        text = re.sub(r'[?]', '?', text)   # Question mark
        text = re.sub(r'[!]', '!', text)   # Exclamation
        
        return text
    
    def _correct_ocr_errors(self, text: str) -> str:
        """Apply OCR error corrections."""
        for wrong, correct in self.ocr_corrections.items():
            text = text.replace(wrong, correct)
        return text
    
    def _remove_headers(self, text: str) -> str:
        """Remove header patterns."""
        # Common header patterns
        patterns = [
            r'(?:পৃষ্ঠা|Page|Pg\.?)\s+\d+',
            r'(?:Chapter|অধ্যায়|পরিচ্ছেদ)\s+\d+',
            r'^\s*[A-Z\s]+$',  # All caps lines
            r'^\s*[\d\s]+$',   # Lines with only numbers
        ]
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            is_header = False
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_header = True
                    break
            if not is_header:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _remove_footers(self, text: str) -> str:
        """Remove footer patterns."""
        # Similar to header removal, check at end of lines
        lines = text.split('\n')
        if len(lines) < 3:
            return text
        
        # Check last few lines for footer patterns
        for i in range(len(lines) - 1, max(0, len(lines) - 3), -1):
            if re.match(r'^\s*\d+\s*$', lines[i]):  # Page numbers
                lines[i] = ''
        
        return '\n'.join(lines)
    
    def _remove_page_numbers(self, text: str) -> str:
        """Remove page number patterns."""
        patterns = [
            r'^\s*\d+\s*$',  # Just numbers
            r'পৃষ্ঠা\s*\d+',  # Bengali page numbers
            r'Page\s*\d+',   # English page numbers
            r'-\s*\d+\s*-',  # Page numbers with dashes
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean excessive whitespace."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([।?!.,;:])', r'\1', text)
        
        # Remove spaces after opening quotes
        text = re.sub(r'"\s+', '"', text)
        
        return text.strip()
    
    def _remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate consecutive lines."""
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                unique_lines.append(line)
                seen.add(line_stripped)
            elif not line_stripped:
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def _normalize_newlines(self, text: str) -> str:
        """Normalize newline characters."""
        # Convert various newline formats
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _preserve_paragraphs(self, text: str) -> str:
        """Preserve paragraph structure."""
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Empty line indicates paragraph break
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)