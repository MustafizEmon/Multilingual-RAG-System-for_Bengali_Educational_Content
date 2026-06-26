import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import json
from app.core.logging import logger
from app.core.config import config

class ShortTermMemory:
    """Short-term memory for conversation context."""
    
    def __init__(self):
        self.config = config.get_section("memory")
        self.max_turns = self.config.get("short_term_turns", 10)
        self.session_timeout = self.config.get("session_timeout_hours", 24)
        
        self.memory = {}  # session_id -> list of turns
    
    def add_turn(self, session_id: str, question: str, answer: str):
        """Add a conversation turn to memory."""
        if session_id not in self.memory:
            self.memory[session_id] = []
        
        self.memory[session_id].append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Limit memory size
        if len(self.memory[session_id]) > self.max_turns:
            self.memory[session_id] = self.memory[session_id][-self.max_turns:]
        
        logger.debug(f"Added turn to session {session_id}", 
                    session_length=len(self.memory[session_id]))
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        if session_id not in self.memory:
            return []
        
        return self.memory[session_id]
    
    def get_recent_turns(self, session_id: str, n: int = 3) -> List[Dict[str, str]]:
        """Get recent conversation turns."""
        history = self.get_history(session_id)
        return history[-n:] if history else []
    
    def clear_session(self, session_id: str):
        """Clear session memory."""
        if session_id in self.memory:
            del self.memory[session_id]
            logger.debug(f"Cleared session {session_id}")


class PersistentMemory:
    """Persistent memory using SQLite database."""
    
    def __init__(self):
        self.config = config.get_section("memory")
        self.db_path = self.config.get("sqlite_path", "./data/memory.db")
        
        self._initialize_db()
        logger.info(f"Persistent memory initialized at {self.db_path}")
    
    def _initialize_db(self):
        """Initialize SQLite database and create tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON conversations(session_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def save_turn(self, session_id: str, question: str, answer: str, 
                  metadata: Optional[Dict[str, Any]] = None):
        """Save a conversation turn to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, user_message, assistant_message, metadata)
            VALUES (?, ?, ?, ?)
        ''', (session_id, question, answer, metadata_json))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Saved turn for session {session_id}")
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row['id'],
                'session_id': row['session_id'],
                'question': row['user_message'],
                'answer': row['assistant_message'],
                'timestamp': row['timestamp'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })
        
        return history
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_turns,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM conversations 
            WHERE session_id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'session_id': session_id,
                'total_turns': row['total_turns'],
                'first_message': row['first_message'],
                'last_message': row['last_message'],
                'duration_minutes': (
                    (datetime.fromisoformat(row['last_message']) - 
                     datetime.fromisoformat(row['first_message'])).total_seconds() / 60
                ) if row['first_message'] and row['last_message'] else 0
            }
        
        return {'session_id': session_id, 'total_turns': 0}
    
    def delete_session(self, session_id: str):
        """Delete all conversations for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM conversations WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Deleted session {session_id}")