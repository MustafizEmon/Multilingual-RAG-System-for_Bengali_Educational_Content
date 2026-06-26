import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import json

class StructuredLogger:
    """Structured logging with JSON output support."""
    
    def __init__(self, name: str, config: dict):
        self.logger = logging.getLogger(name)
        self.config = config
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with handlers and formatters."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create logs directory
        log_file = Path(log_config.get("file_path", "./logs/app.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(log_level)
    
    def log(self, level: str, message: str, **kwargs):
        """Log a structured message with additional context."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            **kwargs
        }
        
        log_method(json.dumps(log_entry, ensure_ascii=False))
    
    def info(self, message: str, **kwargs):
        self.log("info", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log("error", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log("debug", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log("warning", message, **kwargs)

# Create global logger instance
logger = StructuredLogger("rag_system", {})