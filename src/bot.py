import time
from typing import Dict, List
from src.models import Paper
from src.retriever import SemanticScholarRetriever
from src.embeddings import EmbeddingReranker
from src.synthesizer import ClaudeSynthesizer
from src.cache import ResultCache

class EnhancedMedQueryBot:
    def __init__(self, claude_api_key: str, s2_api_key: str, use_cache: bool = True):
        self.retriever = SemanticScholarRetriever(s2_api_key)
        self.reranker = EmbeddingReranker(self.retriever)
        self.synthesizer = ClaudeSynthesizer(claude_api_key)
        self.use_cache = use_cache
        self.cache = ResultCache() if use_cache else None

    def answer_question(self, user_question: str,
                        papers_per_query: int = 15,
                        detect_contradictions: bool = True,
                        top_k_papers: int = 8,
                        use_cached: bool = True) -> Dict:
        """Main method to answer a question with full RAG pipeline"""

        print(f"\nQuestion: {user_question}\n")

        # Step 1: Generate search queries (deterministic from Claude)
        print("    Generating search queries...")
        queries = self.synthesizer.generate_search_queries(user_question)
        print(f"   Queries generated: {len(queries)}")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        print()

        # Step 2: Search for papers with caching
        print("  Searching for papers...")
        all_papers = []
        seen_paper_ids = set()

        for query in queries:
            # Check cache first
            if use_cached and self.cache:
                cached_papers = self.cache.get_cached_papers(query)
                if cached_papers:
                    # Reconstruct Paper objects from cache
                    for paper_dict in cached_papers:
                        if paper_dict['paper_id'] not in seen_paper_ids:
                            paper = Paper(**paper_dict)
                            all_papers.append(paper)
                            seen_paper_ids.add(paper_dict['paper_id'])
                    continue

            # If not cached, fetch from API
            papers = self.retriever.search_papers(query, limit=papers_per_query)

            # Deduplicate
            new_papers = []
            for paper in papers:
                if paper.paper_id and paper.paper_id not in seen_paper_ids:
                    all_papers.append(paper)
                    seen_paper_ids.add(paper.paper_id)
                    new_papers.append(paper)
                elif not paper.paper_id and paper.title not in [p.title for p in all_papers]:
                    # Fallback to title-based deduplication
                    all_papers.append(paper)
                    new_papers.append(paper)

            # Cache the results
            if self.cache and new_papers:
                self.cache.cache_papers(query, new_papers)

            time.sleep(1)  # Rate limiting

        print(f"   Found {len(all_papers)} unique papers\n")

        if not all_papers:
            return {
                "question": user_question,
                "queries_used": queries,
                "papers_analyzed": 0,
                "papers_used": 0,
                "papers": [],
                "contradictions": None,
                "answer": "No papers found. Try rephrasing your question or broadening the scope."
            }

        # Step 3: Rerank using embeddings
        print("Reranking papers by relevance...")
        top_papers = self.reranker.rerank_papers(user_question, all_papers, top_k=top_k_papers)
        print()

        # Step 4: Check if we have a cached answer
        paper_ids = [p.paper_id for p in top_papers if p.paper_id]

        if use_cached and self.cache and paper_ids:
            cached_answer = self.cache.get_cached_answer(user_question, paper_ids)
            if cached_answer:
                print("Using cached answer\n")
                return {
                    "question": user_question,
                    "queries_used": queries,
                    "papers_analyzed": len(all_papers),
                    "papers_used": len(top_papers),
                    "papers": top_papers,
                    "contradictions": None,
                    "answer": cached_answer,
                    "from_cache": True
                }

        # Step 5: Detect contradictions
        contradictions = None
        if detect_contradictions and len(top_papers) >= 2:
            print("  Analyzing for contradictions...")
            contradictions = self.synthesizer.detect_contradictions(top_papers)
            print("   Done\n")

        # Step 6: Synthesize answer
        print("  Synthesizing answer with Claude...")
        answer = self.synthesizer.synthesize_findings(user_question, top_papers, contradictions)
        print("   Done\n")

        # Cache the answer
        if self.cache and paper_ids:
            self.cache.cache_answer(user_question, paper_ids, answer)

        return {
            "question": user_question,
            "queries_used": queries,
            "papers_analyzed": len(all_papers),
            "papers_used": len(top_papers),
            "papers": top_papers,
            "contradictions": contradictions,
            "answer": answer,
            "from_cache": False
        }

    def clear_cache(self):
        """Clear all cached results"""
        if self.cache:
            self.cache.clear_cache()