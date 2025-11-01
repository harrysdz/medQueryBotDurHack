# ============================================================================
# FILE: cache.py (NEW FILE)
# ============================================================================
import json
import hashlib
import os
from typing import Optional, Dict, Any
from pathlib import Path

class ResultCache:
    """Cache search results and answers to ensure consistency"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Separate caches for different components
        self.papers_dir = self.cache_dir / "papers"
        self.answers_dir = self.cache_dir / "answers"

        self.papers_dir.mkdir(exist_ok=True)
        self.answers_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()

    def get_cached_papers(self, query: str) -> Optional[list]:
        """Retrieve cached papers for a query"""
        cache_key = self._get_cache_key(query)
        cache_file = self.papers_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    print(f"   💾 Using cached papers for query: '{query[:50]}...'")
                    return data['papers']
            except Exception as e:
                print(f"   ⚠️  Cache read error: {e}")
                return None

        return None

    def cache_papers(self, query: str, papers: list):
        """Cache papers for a query"""
        cache_key = self._get_cache_key(query)
        cache_file = self.papers_dir / f"{cache_key}.json"

        try:
            # Convert papers to serializable format
            papers_data = []
            for paper in papers:
                paper_dict = {
                    'title': paper.title,
                    'abstract': paper.abstract,
                    'year': paper.year,
                    'authors': paper.authors,
                    'citation_count': paper.citation_count,
                    'url': paper.url,
                    'tldr': paper.tldr,
                    'relevance_score': paper.relevance_score,
                    'paper_id': paper.paper_id
                }
                papers_data.append(paper_dict)

            with open(cache_file, 'w') as f:
                json.dump({
                    'query': query,
                    'papers': papers_data
                }, f, indent=2)

            print(f"   💾 Cached {len(papers)} papers")

        except Exception as e:
            print(f"   ⚠️  Cache write error: {e}")

    def get_cached_answer(self, question: str, paper_ids: list) -> Optional[str]:
        """Retrieve cached answer for question + papers combination"""
        # Create cache key from question and paper IDs
        cache_input = f"{question}|{'|'.join(sorted(paper_ids))}"
        cache_key = self._get_cache_key(cache_input)
        cache_file = self.answers_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    print(f"   💾 Using cached answer")
                    return data['answer']
            except Exception as e:
                print(f"   ⚠️  Cache read error: {e}")
                return None

        return None

    def cache_answer(self, question: str, paper_ids: list, answer: str):
        """Cache an answer"""
        cache_input = f"{question}|{'|'.join(sorted(paper_ids))}"
        cache_key = self._get_cache_key(cache_input)
        cache_file = self.answers_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'question': question,
                    'paper_ids': paper_ids,
                    'answer': answer
                }, f, indent=2)

            print(f"   💾 Cached answer")

        except Exception as e:
            print(f"   ⚠️  Cache write error: {e}")

    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.__init__(str(self.cache_dir))
        print("   🗑️  Cache cleared")