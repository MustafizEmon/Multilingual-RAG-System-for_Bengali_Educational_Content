import yaml
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration manager for the RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Groq API
        if os.getenv("GROQ_API_KEY"):
            self.config["llm"]["groq_api_key"] = os.getenv("GROQ_API_KEY")
        
        # Database paths
        if os.getenv("VECTOR_DB_PATH"):
            self.config["vectorstore"]["persist_directory"] = os.getenv("VECTOR_DB_PATH")
        
        if os.getenv("SQLITE_PATH"):
            self.config["memory"]["sqlite_path"] = os.getenv("SQLITE_PATH")
        
        # Model overrides
        if os.getenv("EMBEDDING_MODEL"):
            self.config["embedding"]["model_name"] = os.getenv("EMBEDDING_MODEL")
        
        if os.getenv("LLM_MODEL"):
            self.config["llm"]["primary_model"] = os.getenv("LLM_MODEL")
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def __getattr__(self, name):
        """Allow attribute-style access to config sections."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no section '{name}'")

# Singleton config instance
config = Config()