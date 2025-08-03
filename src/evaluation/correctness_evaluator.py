"""
Correctness evaluation system for RAG responses and system improvements.
"""
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from collections import Counter
import logging
import openai
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

class CorrectnessEvaluator:
    """Evaluates the correctness of RAG responses and system improvements"""
    
    def __init__(self):
        self.metrics_history = []
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.evaluation_criteria = {
            "correctness": 0.35,        # Factual accuracy grounded in documents
            "rouge_score": 0.25,        # ROUGE evaluation
            "ndcg_score": 0.25,         # Ranking quality
            "code_quality": 0.15        # Generated code quality
        }
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def evaluate_iteration(
        self, 
        query: str,
        retrieved_chunks: List[Dict],
        generated_response: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate correctness after each RAG iteration using ROUGE, nDCG, and factual accuracy
        
        Returns:
            Dictionary containing metric scores and overall evaluation
        """
        # 1. Evaluate Correctness (Factual accuracy grounded in documents)
        correctness_score = await self._evaluate_correctness(
            generated_response, retrieved_chunks
        )
        
        # 2. Calculate ROUGE scores
        rouge_scores = self._calculate_rouge_scores(
            generated_response, retrieved_chunks, ground_truth
        )
        
        # 3. Calculate nDCG (Normalized Discounted Cumulative Gain)
        ndcg_score = self._calculate_ndcg(
            query, retrieved_chunks, generated_response
        )
        
        # 4. Evaluate code quality if applicable
        code_score = 1.0
        if self._contains_code(generated_response):
            code_score = await self._evaluate_code_quality(generated_response)
        
        # Calculate weighted overall score
        overall_score = (
            self.evaluation_criteria["correctness"] * correctness_score +
            self.evaluation_criteria["rouge_score"] * rouge_scores["average"] +
            self.evaluation_criteria["ndcg_score"] * ndcg_score +
            self.evaluation_criteria["code_quality"] * code_score
        )
        
        # Store metrics
        iteration_metrics = {
            "timestamp": datetime.now(),
            "overall_score": overall_score,
            "correctness": correctness_score,
            "rouge_1": rouge_scores["rouge1"],
            "rouge_2": rouge_scores["rouge2"],
            "rouge_l": rouge_scores["rougeL"],
            "ndcg": ndcg_score,
            "code_quality": code_score,
            "query": query
        }
        
        self.metrics_history.append(iteration_metrics)
        return self._generate_metrics_report(iteration_metrics)
    
    async def _evaluate_correctness(
        self, 
        response: str, 
        retrieved_chunks: List[Dict]
    ) -> float:
        """
        Evaluate factual accuracy by checking if response is grounded in retrieved documents
        """
        if not retrieved_chunks:
            return 0.0
        
        # Extract facts from response
        response_facts = self._extract_facts(response)
        
        # Combine all retrieved content
        retrieved_content = ' '.join([chunk.get('content', '') for chunk in retrieved_chunks])
        retrieved_facts = self._extract_facts(retrieved_content)
        
        # Calculate fact coverage (how many facts in response are supported by documents)
        supported_facts = 0
        total_facts = len(response_facts)
        
        for fact in response_facts:
            if self._is_fact_supported(fact, retrieved_facts, retrieved_content):
                supported_facts += 1
        
        # Penalize hallucination (facts not in retrieved documents)
        correctness = supported_facts / total_facts if total_facts > 0 else 0.0
        
        # Verify no contradictions
        contradiction_penalty = self._check_contradictions(response_facts, retrieved_facts)
        
        return max(0.0, correctness - contradiction_penalty)
    
    def _calculate_rouge_scores(
        self, 
        response: str, 
        retrieved_chunks: List[Dict],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for response evaluation
        """
        # Use ground truth if available, otherwise use retrieved content
        reference = ground_truth if ground_truth else ' '.join([c.get('content', '') for c in retrieved_chunks])
        
        scores = self.rouge_scorer.score(reference, response)
        
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
            "average": np.mean([
                scores['rouge1'].fmeasure,
                scores['rouge2'].fmeasure,
                scores['rougeL'].fmeasure
            ])
        }
    
    def _calculate_ndcg(
        self, 
        query: str, 
        retrieved_chunks: List[Dict],
        response: str
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain for ranking quality
        """
        if not retrieved_chunks:
            return 0.0
        
        # Calculate relevance scores for each chunk
        relevance_scores = []
        for i, chunk in enumerate(retrieved_chunks):
            # Relevance based on: query similarity, response usage, and position
            query_similarity = self._calculate_similarity(query, chunk.get('content', ''))
            usage_score = self._chunk_usage_in_response(chunk.get('content', ''), response)
            
            # Combine scores (you can adjust weights)
            relevance = 0.6 * query_similarity + 0.4 * usage_score
            relevance_scores.append(relevance)
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Calculate ideal DCG (if chunks were perfectly ordered)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Normalize
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def _extract_facts(self, text: str) -> Set[str]:
        """Extract factual statements from text"""
        # Simplified fact extraction - in production, use NLP models
        sentences = nltk.sent_tokenize(text)
        facts = set()
        
        for sent in sentences:
            # Extract noun phrases and named entities as facts
            tokens = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            
            # Extract noun phrases
            noun_phrases = []
            current_np = []
            for word, pos in pos_tags:
                if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']:
                    current_np.append(word)
                elif current_np:
                    if len(current_np) > 1:
                        noun_phrases.append(' '.join(current_np))
                    current_np = []
            
            if current_np and len(current_np) > 1:
                noun_phrases.append(' '.join(current_np))
            
            facts.update(noun_phrases)
        
        return facts
    
    def _is_fact_supported(self, fact: str, document_facts: Set[str], document_text: str) -> bool:
        """Check if a fact is supported by the retrieved documents"""
        # Direct match
        if fact in document_facts:
            return True
        
        # Fuzzy match in document text
        fact_lower = fact.lower()
        if fact_lower in document_text.lower():
            return True
        
        # Semantic similarity check (simplified)
        for doc_fact in document_facts:
            similarity = self._calculate_similarity(fact, doc_fact)
            if similarity > 0.8:  # Threshold for semantic match
                return True
        
        return False
    
    def _check_contradictions(self, response_facts: Set[str], document_facts: Set[str]) -> float:
        """Check for contradictions between response and documents"""
        # Simplified implementation - in production, use more sophisticated NLI models
        contradiction_penalty = 0.0
        
        # Check for negation patterns
        negation_words = ['not', 'no', 'never', 'none', 'neither', 'nor']
        
        for fact in response_facts:
            fact_words = fact.lower().split()
            has_negation = any(neg in fact_words for neg in negation_words)
            
            # If the fact contains negation, check if the positive version exists in documents
            if has_negation:
                # Remove negation words and check
                positive_fact = ' '.join([w for w in fact_words if w not in negation_words])
                if any(positive_fact in doc_fact.lower() for doc_fact in document_facts):
                    contradiction_penalty += 0.1
        
        return min(contradiction_penalty, 0.5)  # Cap penalty at 0.5
    
    def _contains_code(self, text: str) -> bool:
        """Check if the text contains code snippets"""
        code_indicators = ['```', 'def ', 'class ', 'import ', 'from ', 'async def', 'function', 'const ']
        return any(indicator in text for indicator in code_indicators)
    
    async def _evaluate_code_quality(self, text: str) -> float:
        """Evaluate the quality of code in the response"""
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        
        if not code_blocks:
            return 1.0
        
        total_score = 0.0
        for code in code_blocks:
            # Check for common quality indicators
            score = 1.0
            
            # Deduct for missing docstrings
            if 'def ' in code and '"""' not in code:
                score -= 0.1
            
            # Deduct for no error handling
            if 'try:' not in code and 'except' not in code:
                score -= 0.1
            
            # Deduct for very short code (likely incomplete)
            if len(code.strip()) < 50:
                score -= 0.2
            
            # Bonus for type hints
            if '->' in code or ': ' in code:
                score += 0.1
            
            total_score += max(0.0, min(1.0, score))
        
        return total_score / len(code_blocks) if code_blocks else 1.0
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Simplified implementation using word overlap
        # In production, use proper embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _chunk_usage_in_response(self, chunk_content: str, response: str) -> float:
        """Calculate how much a chunk was used in generating the response"""
        chunk_words = set(chunk_content.lower().split())
        response_words = set(response.lower().split())
        
        if not chunk_words:
            return 0.0
        
        overlap = chunk_words.intersection(response_words)
        return len(overlap) / len(chunk_words)
    
    def _generate_metrics_report(self, metrics: Dict) -> Dict[str, Any]:
        """Generate comprehensive user-facing metrics report"""
        return {
            "overall_score": round(metrics["overall_score"] * 100, 2),
            "evaluation_metrics": {
                "correctness": {
                    "score": round(metrics["correctness"] * 100, 2),
                    "description": "Factual accuracy grounded in retrieved documents"
                },
                "rouge_scores": {
                    "rouge_1": round(metrics["rouge_1"] * 100, 2),
                    "rouge_2": round(metrics["rouge_2"] * 100, 2),
                    "rouge_l": round(metrics["rouge_l"] * 100, 2),
                    "description": "Text overlap with reference content"
                },
                "ndcg": {
                    "score": round(metrics["ndcg"] * 100, 2),
                    "description": "Quality of document ranking"
                },
                "code_quality": {
                    "score": round(metrics["code_quality"] * 100, 2),
                    "description": "Quality of generated code (if applicable)"
                }
            },
            "improvement_trend": self._calculate_improvement_trend()
        }
    
    def _calculate_improvement_trend(self) -> float:
        """Calculate improvement trend over last 5 iterations"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = self.metrics_history[-5:]
        correctness_values = [m["correctness"] for m in recent_metrics]
        
        # Calculate trend (positive means improvement)
        if len(correctness_values) >= 2:
            trend = (correctness_values[-1] - correctness_values[0]) / correctness_values[0] if correctness_values[0] > 0 else 0
            return round(trend * 100, 2)
        return 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation metrics"""
        if not self.metrics_history:
            return {"message": "No evaluations performed yet"}
        
        latest = self.metrics_history[-1]
        
        return {
            "latest_evaluation": {
                "timestamp": latest["timestamp"].isoformat(),
                "overall_score": round(latest["overall_score"] * 100, 2),
                "correctness": round(latest["correctness"] * 100, 2),
                "ndcg": round(latest["ndcg"] * 100, 2)
            },
            "history": {
                "total_evaluations": len(self.metrics_history),
                "average_score": round(np.mean([m["overall_score"] for m in self.metrics_history]) * 100, 2),
                "improvement_trend": self._calculate_improvement_trend()
            }
        }