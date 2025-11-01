import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from src.models import Paper
from src.retriever import SemanticScholarRetriever

class EmbeddingReranker:
    def __init__(self, retriever: SemanticScholarRetriever):
        self.retriever = retriever

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Check for invalid values
        if np.isnan(v1).any() or np.isnan(v2).any():
            return 0.0

        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def rerank_papers(self, query: str, papers: List[Paper], top_k: int = 10) -> List[Paper]:
        """Rerank papers using semantic similarity"""
        if not papers:
            return []

        # Get embeddings for query and papers
        query_embedding = self.retriever.get_embedding(query)

        paper_texts = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract[:500]}"
            paper_texts.append(text)

        paper_embeddings = []
        for paper_text in paper_texts:
            embedding = self.retriever.get_embedding(paper_text)
            paper_embeddings.append(embedding)
        # Calculate relevance scores
        for paper, embedding in zip(papers, paper_embeddings):
            paper.embedding = embedding
            similarity = self._cosine_similarity(query_embedding, embedding)
            # Combine semantic similarity with citation count and recency
            citation_score = min(paper.citation_count / 100, 1.0)
            recency_score = (paper.year - 2020) / 10 if paper.year >= 2020 else 0
            paper.relevance_score = (0.6 * similarity +
                                     0.25 * citation_score +
                                     0.15 * recency_score)

        # Sort by relevance score
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers[:top_k]
