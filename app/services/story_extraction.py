import re
from typing import str, Optional, Dict, Any
from app.core.logging import logger
from app.core.config import config

class StoryExtractionService:
    """Extract story content from documents using markers and patterns."""
    
    def __init__(self):
        self.config = config.get_section("story_extraction")
        self.start_marker = self.config.get("start_marker", "গল্প শুরু")
        self.end_marker = self.config.get("end_marker", "গল্প শেষ")
        self.fallback_regex = self.config.get("fallback_regex", "গল্প.*?(?:[\n]{2,}|$)")
    
    def extract_story(self, text: str) -> Dict[str, Any]:
        """Extract story content from text."""
        if not text:
            return {"story": "", "method": "none", "confidence": 0.0}
        
        logger.debug("Starting story extraction", text_length=len(text))
        
        # Try marker-based extraction first
        story, confidence, method = self._extract_with_markers(text)
        
        # Fallback to regex
        if not story or confidence < 0.5:
            story, confidence = self._extract_with_regex(text)
            method = "regex"
        
        # Final fallback: take first section
        if not story or confidence < 0.3:
            story, confidence = self._extract_by_section(text)
            method = "section_detection"
        
        logger.debug("Story extraction complete", 
                    method=method,
                    confidence=confidence,
                    story_length=len(story))
        
        return {
            "story": story,
            "method": method,
            "confidence": confidence
        }
    
    def _extract_with_markers(self, text: str) -> tuple:
        """Extract content between start and end markers."""
        start_idx = text.find(self.start_marker)
        if start_idx == -1:
            return "", 0.0, "markers"
        
        end_idx = text.find(self.end_marker, start_idx)
        if end_idx == -1:
            # If end marker not found, take content until end
            story = text[start_idx + len(self.start_marker):].strip()
            confidence = 0.7
        else:
            story = text[start_idx + len(self.start_marker):end_idx].strip()
            confidence = 0.95
        
        return story, confidence, "markers"
    
    def _extract_with_regex(self, text: str) -> tuple:
        """Extract using regular expression patterns."""
        try:
            match = re.search(self.fallback_regex, text, re.DOTALL | re.IGNORECASE)
            if match:
                story = match.group(0).strip()
                # Check if story has substantial content
                if len(story) > 50:
                    confidence = 0.7
                else:
                    confidence = 0.3
                return story, confidence
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
        
        return "", 0.0
    
    def _extract_by_section(self, text: str) -> tuple:
        """Extract by identifying section boundaries."""
        # Look for common section patterns in Bengali
        section_patterns = [
            r'[অধ্যায়|পরিচ্ছেদ|কাণ্ড]\s+[০-৯]+\s*[ঃ]',  # Chapter indicators
            r'[১-৯]\.[\d]*\s+',  # Numbered sections
            r'[গল্প|কথা|উপন্যাস]',  # Story indicators
        ]
        
        lines = text.split('\n')
        start_idx = 0
        
        # Find first line matching a section pattern
        for i, line in enumerate(lines):
            if any(re.search(pattern, line) for pattern in section_patterns):
                start_idx = i
                break
        
        story = '\n'.join(lines[start_idx:])
        confidence = 0.4 if len(story) > 100 else 0.2
        
        return story, confidence