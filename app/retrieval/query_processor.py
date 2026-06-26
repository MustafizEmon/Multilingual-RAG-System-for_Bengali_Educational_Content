from typing import str, List, Dict, Any, Optional
import re
from app.core.logging import logger
from app.core.config import config
from app.services.memory import ShortTermMemory

class QueryProcessor:
    """Query processing and rewriting for conversational context."""
    
    def __init__(self, short_term_memory: ShortTermMemory):
        self.memory = short_term_memory
        self.config = config.get_section("retrieval")
    
    def process_query(self, query: str, session_id: str) -> str:
        """Process query with context awareness."""
        logger.debug(f"Processing query: {query[:100]}...")
        
        # Get conversation history
        history = self.memory.get_history(session_id)
        
        # Rewrite query if needed
        if len(history) > 0:
            rewritten_query = self._rewrite_query(query, history)
            if rewritten_query != query:
                logger.debug(f"Query rewritten: {query} -> {rewritten_query}")
                return rewritten_query
        
        return query
    
    def _rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """Rewrite query to resolve references."""
        # Check if query is a follow-up
        if not self._is_follow_up(query):
            return query
        
        # Try to resolve references
        context = self._extract_context(history)
        
        # Handle Bengali pronouns
        if self._is_bengali(query):
            return self._rewrite_bengali(query, context)
        else:
            return self._rewrite_english(query, context)
    
    def _is_follow_up(self, query: str) -> bool:
        """Check if query is likely a follow-up."""
        # Check for pronouns and references
        followup_patterns = [
            r'\b(তিনি|সে|এটি|এটা|ওই|এই|ওরা|তারা)\b',  # Bengali pronouns
            r'\b(he|she|it|they|this|that|these|those|the)\b',  # English pronouns
            r'^(এতে|এখানে|সেখানে)',  # Bengali references
            r'^(there|here|then|so|thus)'  # English references
        ]
        
        for pattern in followup_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _is_bengali(self, text: str) -> bool:
        """Check if text contains Bengali characters."""
        return any('\u0980' <= c <= '\u09FF' for c in text)
    
    def _extract_context(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract context from conversation history."""
        if not history:
            return {}
        
        # Get the last few turns
        recent_turns = history[-3:]  # Last 3 turns
        
        context = {
            'last_question': recent_turns[-1]['question'] if recent_turns else '',
            'last_answer': recent_turns[-1]['answer'] if recent_turns else '',
            'entities': [],
            'topics': []
        }
        
        # Extract entities (simple approach)
        for turn in recent_turns:
            # Look for named entities (capitalized words in English)
            if not self._is_bengali(turn['answer']):
                entities = re.findall(r'\b[A-Z][a-z]+\b', turn['answer'])
                context['entities'].extend(entities)
            
            # Extract key topics (words that appear frequently)
            words = turn['answer'].split()
            if len(words) > 5:
                # Simple topic extraction: nouns (crude approximation)
                for word in words:
                    if len(word) > 3:
                        context['topics'].append(word)
        
        # Deduplicate and limit
        context['entities'] = list(set(context['entities']))[:5]
        context['topics'] = list(set(context['topics']))[:10]
        
        return context
    
    def _rewrite_bengali(self, query: str, context: Dict[str, Any]) -> str:
        """Rewrite Bengali query with context."""
        rewritten = query
        
        # Simple pronoun resolution
        pronoun_map = {
            'তিনি': context.get('entities', [''])[0] if context.get('entities') else 'তিনি',
            'সে': context.get('entities', [''])[0] if context.get('entities') else 'সে',
            'এটি': context.get('topics', [''])[0] if context.get('topics') else 'এটি',
            'ওই': context.get('topics', [''])[0] if context.get('topics') else 'ওই'
        }
        
        for pronoun, replacement in pronoun_map.items():
            if pronoun in query and replacement:
                rewritten = rewritten.replace(pronoun, replacement)
        
        return rewritten
    
    def _rewrite_english(self, query: str, context: Dict[str, Any]) -> str:
        """Rewrite English query with context."""
        rewritten = query
        
        # Simple pronoun resolution
        pronoun_map = {
            'he': context.get('entities', [''])[0] if context.get('entities') else 'he',
            'she': context.get('entities', [''])[0] if context.get('entities') else 'she',
            'it': context.get('topics', [''])[0] if context.get('topics') else 'it',
            'they': context.get('entities', [''])[0] if context.get('entities') else 'they',
            'this': context.get('topics', [''])[0] if context.get('topics') else 'this'
        }
        
        # Use regex to replace whole words only
        for pronoun, replacement in pronoun_map.items():
            if replacement and replacement != pronoun:
                pattern = r'\b' + pronoun + r'\b'
                rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten