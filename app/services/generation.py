from groq import Groq
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timezone
import json
from app.core.logging import logger
from app.core.config import config

class GenerationService:
    """LLM generation service with prompt engineering and inference."""
    
    def __init__(self):
        self.config = config.get_section("llm")
        self.provider = self.config.get("provider", "groq")
        self.api_key = self.config.get("groq_api_key")
        self.primary_model = self.config.get("primary_model", "openai/gpt-oss-120b")
        self.fallback_model = self.config.get("fallback_model", "qwen/qwen3.6-27b")
        self.final_fallback = self.config.get("final_fallback", "openai/gpt-oss-20b")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 512)
        self.top_p = self.config.get("top_p", 0.9)
        self.async_inference = self.config.get("async_inference", True)
        
        # Initialize Groq client
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model {self.primary_model}")
        else:
            logger.warning("No Groq API key found. LLM generation will not work.")
    
    async def generate_response(self, query: str, context: List[Dict[str, Any]], 
                                session_id: str, language: str = 'bn') -> Dict[str, Any]:
        """Generate response using RAG."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Prepare prompt
            prompt = self._create_prompt(query, context, language)
            
            # Get response from LLM
            if self.async_inference:
                response = await self._async_completion(prompt)
            else:
                response = self._sync_completion(prompt)
            
            # Parse response
            result = self._parse_response(response, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(result, context)
            
            # Extract citations
            citations = self._extract_citations(result, context)
            
            # Log generation metrics
            end_time = datetime.now(timezone.utc)
            generation_time = (end_time - start_time).total_seconds()
            logger.info(f"Generated response in {generation_time:.2f}s",
                       session_id=session_id,
                       language=language,
                       confidence=confidence)
            
            return {
                'answer': result.get('answer', ''),
                'confidence': confidence,
                'citations': citations,
                'model': self.primary_model,
                'generation_time': generation_time,
                'tokens_used': len(prompt) // 4 + len(result.get('answer', '')) // 4
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'answer': 'উত্তর পাওয়া যায়নি' if language == 'bn' else 'Answer not found',
                'confidence': 0.0,
                'citations': [],
                'error': str(e)
            }
    
    def _create_prompt(self, query: str, context: List[Dict[str, Any]], 
                       language: str) -> str:
        """Create production-ready RAG prompt."""
        # Prepare context with citations
        context_text = ""
        for i, doc in enumerate(context):
            chunk_id = doc.get('chunk_id', f'chunk_{i}')
            context_text += f"[{chunk_id}] {doc['text']}\n\n"
        
        # System instruction based on language
        if language == 'bn':
            system_instruction = """আপনি একজন বাংলা শিক্ষা সহকারী। আপনার কাজ হল:
1. শুধুমাত্র প্রদত্ত প্রসঙ্গ থেকে উত্তর দিন
2. কখনও বানোয়াট তথ্য তৈরি করবেন না
3. বাহ্যিক জ্ঞান ব্যবহার করবেন না
4. প্রতিটি উদ্ধৃতির জন্য চাঙ্ক আইডি উল্লেখ করুন
5. সংক্ষিপ্ত এবং স্পষ্ট উত্তর দিন
6. যদি উত্তর না পাওয়া যায়, তাহলে "উত্তর পাওয়া যায়নি" বলুন
7. প্রশ্নের ভাষায় উত্তর দিন

প্রসঙ্গ:
{context}

প্রশ্ন: {query}

উত্তর:"""
        else:
            system_instruction = """You are an English educational assistant. Your rules:
1. Answer using ONLY the provided context
2. Never fabricate information
3. Never use external knowledge
4. Cite chunk IDs for each reference
5. Be concise and clear
6. If answer not found, say "Answer not found"
7. Answer in the same language as the question

Context:
{context}

Question: {query}

Answer:"""
        
        prompt = system_instruction.format(
            context=context_text,
            query=query
        )
        
        return prompt
    
    async def _async_completion(self, prompt: str) -> str:
        """Asynchronous completion using Groq."""
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.primary_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, trying fallback")
            return await self._async_fallback_completion(prompt)
    
    async def _async_fallback_completion(self, prompt: str) -> str:
        """Try fallback models."""
        models = [self.fallback_model, self.final_fallback]
        
        for model in models:
            try:
                completion = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )
                logger.info(f"Used fallback model: {model}")
                return completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        raise RuntimeError("All LLM models failed")
    
    def _sync_completion(self, prompt: str) -> str:
        """Synchronous completion (for non-async contexts)."""
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        try:
            completion = self.client.chat.completions.create(
                model=self.primary_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, trying fallback")
            return self._sync_fallback_completion(prompt)
    
    def _sync_fallback_completion(self, prompt: str) -> str:
        """Synchronous fallback completion."""
        models = [self.fallback_model, self.final_fallback]
        
        for model in models:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )
                logger.info(f"Used fallback model: {model}")
                return completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        raise RuntimeError("All LLM models failed")
    
    def _parse_response(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM response and extract structured information."""
        # Clean response
        response = response.strip()
        
        # Extract citations if present
        citations = []
        import re
        citation_pattern = r'\[(chunk_\d+)\]'
        matches = re.findall(citation_pattern, response)
        citations.extend(matches)
        
        # Check if answer is a refusal
        if 'উত্তর পাওয়া যায়নি' in response or 'Answer not found' in response:
            return {
                'answer': response,
                'found': False,
                'citations': citations
            }
        
        return {
            'answer': response,
            'found': True,
            'citations': citations
        }
    
    def _calculate_confidence(self, result: Dict[str, Any], 
                             context: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the response."""
        if not result.get('found', False):
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Boost if citations are present
        if result.get('citations'):
            confidence += 0.15
        
        # Boost if answer is concise and relevant
        answer = result.get('answer', '')
        if len(answer) < 1000 and len(answer) > 10:
            confidence += 0.05
        
        # Check if answer contains proper references to context
        context_indicators = ['উল্লেখ', 'অনুযায়ী', 'according', 'based on']
        if any(indicator in answer.lower() for indicator in context_indicators):
            confidence += 0.05
        
        # Ensure confidence doesn't exceed 1.0
        return min(1.0, confidence)
    
    def _extract_citations(self, result: Dict[str, Any], 
                          context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source citations from response."""
        citations = []
        
        # Get cited chunk IDs
        cited_ids = result.get('citations', [])
        
        # Find corresponding chunks
        for chunk_id in cited_ids:
            for doc in context:
                if doc.get('chunk_id') == chunk_id:
                    citations.append({
                        'chunk_id': chunk_id,
                        'page': doc.get('page', 0),
                        'source': doc.get('source', ''),
                        'section': doc.get('section', ''),
                        'text_preview': doc['text'][:100] + '...' if len(doc['text']) > 100 else doc['text']
                    })
                    break
        
        return citations