import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from app.core.logging import logger
from app.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from app.services.generation import GenerationService
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import VectorStore
from app.services.reranking import RerankingService
from app.services.compression import ContextCompressionService

class Evaluator:
    """Main evaluation framework."""
    
    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        
        # Initialize services
        self.vector_store = VectorStore()
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.reranking_service = RerankingService()
        self.compression_service = ContextCompressionService(self.reranking_service)
        self.generation_service = GenerationService()
    
    def evaluate_retrieval(self, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval performance."""
        logger.info(f"Evaluating retrieval on {len(test_questions)} questions")
        
        results = {
            'recall@5': [], 'precision@5': [], 'mrr': [], 'ndcg@5': [],
            'latency': []
        }
        
        for test in test_questions:
            query = test['question']
            relevant_docs = test.get('relevant_docs', [])
            
            # Time retrieval
            import time
            start = time.time()
            
            # Dense only
            dense_results = self.vector_store.search(query, 20)
            dense_docs = [r.get('chunk_id', '') for r in dense_results]
            
            # Hybrid retrieval
            hybrid_results = self.hybrid_retriever.retrieve(query, 20)
            hybrid_docs = [r.get('chunk_id', '') for r in hybrid_results]
            
            # Hybrid + reranking
            reranked_results = self.reranking_service.rerank(query, hybrid_results)
            reranked_docs = [r.get('chunk_id', '') for r in reranked_results]
            
            latency = time.time() - start
            
            # Calculate metrics for each method
            methods = {
                'dense': dense_docs,
                'hybrid': hybrid_docs,
                'hybrid_reranked': reranked_docs
            }
            
            for method, docs in methods.items():
                metrics = {
                    'recall@5': self.retrieval_metrics.recall_at_k(
                        relevant_docs, docs, 5
                    ),
                    'precision@5': self.retrieval_metrics.precision_at_k(
                        relevant_docs, docs, 5
                    ),
                    'mrr': self.retrieval_metrics.mrr(relevant_docs, docs),
                    'ndcg@5': self.retrieval_metrics.ndcg(
                        relevant_docs, docs, 5
                    )
                }
                
                for metric, value in metrics.items():
                    results[f'{metric}_{method}'] = results.get(
                        f'{metric}_{method}', []
                    )
                    results[f'{metric}_{method}'].append(value)
            
            results['latency'].append(latency)
        
        # Compute averages
        summary = {
            'total_questions': len(test_questions),
            'avg_latency': np.mean(results['latency'])
        }
        
        for key in results:
            if key != 'latency':
                summary[key] = np.mean(results[key])
        
        return summary
    
    def evaluate_generation(self, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate generation performance."""
        logger.info(f"Evaluating generation on {len(test_questions)} questions")
        
        results = {
            'exact_match': [], 'f1_score': [], 'groundedness': [],
            'hallucination_rate': [], 'latency': []
        }
        
        for test in test_questions:
            query = test['question']
            expected_answer = test.get('expected_answer', '')
            context = test.get('context', [])
            
            # Get retrieval context
            import time
            start = time.time()
            
            # Retrieve documents
            retrieved = self.hybrid_retriever.retrieve(query, 10)
            reranked = self.reranking_service.rerank(query, retrieved)
            compressed = self.compression_service.compress(query, reranked)
            
            # Generate answer
            import asyncio
            response = asyncio.run(
                self.generation_service.generate_response(
                    query, compressed, session_id='eval', language='bn'
                )
            )
            
            latency = time.time() - start
            
            # Calculate metrics
            predicted_answer = response.get('answer', '')
            
            results['exact_match'].append(
                self.generation_metrics.exact_match(predicted_answer, expected_answer)
            )
            results['f1_score'].append(
                self.generation_metrics.f1_score(predicted_answer, expected_answer)
            )
            results['groundedness'].append(
                self.generation_metrics.groundedness(
                    predicted_answer, 
                    [doc['text'] for doc in compressed]
                )
            )
            results['hallucination_rate'].append(
                self.generation_metrics.hallucination_rate(
                    predicted_answer,
                    [doc['text'] for doc in compressed]
                )
            )
            results['latency'].append(latency)
        
        # Compute averages
        summary = {
            'total_questions': len(test_questions),
            'avg_latency': np.mean(results['latency'])
        }
        
        for key in results:
            if key != 'latency':
                summary[key] = np.mean(results[key])
        
        return summary
    
    def generate_report(self, test_suite_path: str, output_dir: str) -> Dict[str, Any]:
        """Generate complete evaluation report."""
        # Load test suite
        with open(test_suite_path, 'r', encoding='utf-8') as f:
            test_suite = json.load(f)
        
        # Run evaluations
        retrieval_results = self.evaluate_retrieval(test_suite)
        generation_results = self.evaluate_generation(test_suite)
        
        # Prepare report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'test_suite_size': len(test_suite),
            'retrieval_metrics': retrieval_results,
            'generation_metrics': generation_results,
            'summary': {
                'retrieval_score': np.mean([
                    retrieval_results.get('recall@5_hybrid_reranked', 0),
                    retrieval_results.get('precision@5_hybrid_reranked', 0),
                    retrieval_results.get('mrr_hybrid_reranked', 0)
                ]),
                'generation_score': np.mean([
                    generation_results.get('f1_score', 0),
                    generation_results.get('groundedness', 0)
                ]),
                'overall_score': np.mean([
                    retrieval_results.get('recall@5_hybrid_reranked', 0) * 0.4,
                    generation_results.get('f1_score', 0) * 0.3,
                    generation_results.get('groundedness', 0) * 0.3
                ])
            }
        }
        
        # Save report
        output_path = Path(output_dir) / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate visualizations
        self._generate_visualizations(report, output_dir)
        
        logger.info(f"Evaluation report generated at {output_path}")
        return report
    
    def _generate_visualizations(self, report: Dict[str, Any], output_dir: str):
        """Generate visualizations for the report."""
        # Prepare data
        metrics = report['retrieval_metrics']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Recall@5 comparison
        methods = ['dense', 'hybrid', 'hybrid_reranked']
        recall_values = [metrics.get(f'recall@5_{m}', 0) for m in methods]
        axes[0, 0].bar(methods, recall_values)
        axes[0, 0].set_title('Recall@5 Comparison')
        axes[0, 0].set_ylabel('Recall@5')
        
        # Precision@5 comparison
        precision_values = [metrics.get(f'precision@5_{m}', 0) for m in methods]
        axes[0, 1].bar(methods, precision_values)
        axes[0, 1].set_title('Precision@5 Comparison')
        axes[0, 1].set_ylabel('Precision@5')
        
        # MRR comparison
        mrr_values = [metrics.get(f'mrr_{m}', 0) for m in methods]
        axes[1, 0].bar(methods, mrr_values)
        axes[1, 0].set_title('MRR Comparison')
        axes[1, 0].set_ylabel('MRR')
        
        # Overall metrics
        gen_metrics = report['generation_metrics']
        gen_keys = ['exact_match', 'f1_score', 'groundedness', 'hallucination_rate']
        gen_values = [gen_metrics.get(k, 0) for k in gen_keys]
        axes[1, 1].bar(gen_keys, gen_values)
        axes[1, 1].set_title('Generation Metrics')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir) / f"evaluation_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved at {output_path}")