import requests
import time
import hashlib
from typing import List, Optional
from src.models import Paper

class SemanticScholarRetriever:
    def __init__(self, s2_api_key: str):
        self.s2_api_key = s2_api_key
        self.s2_base_url = "https://api.semanticscholar.org/graph/v1"
        # Cache for embeddings to avoid re-computing
        self._embedding_cache = {}

        # Initialize local embedding model
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            print(" Loading embedding model (this may take a moment)...")
            # Using a small, fast model optimized for semantic search
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(" Embedding model loaded successfully\n")
        except ImportError:
            print("️  sentence-transformers not installed. Run: pip install sentence-transformers")
            print("   Falling back to basic embeddings\n")
        except Exception as e:
            print(f"  Error loading embedding model: {e}")
            print("   Falling back to basic embeddings\n")

    def search_papers(self, query: str, limit: int = 20, year_filter: str = "2020-") -> List[Paper]:
        """Search Semantic Scholar for relevant papers with consistent ordering"""
        headers = {"x-api-key": self.s2_api_key}

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,authors,citationCount,url,tldr,paperId",
            "year": year_filter
        }

        try:
            response = requests.get(
                f"{self.s2_base_url}/paper/search",
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                print(f"  Semantic Scholar API returned status {response.status_code}")
                return []

            data = response.json()
            papers = []

            for item in data.get("data", []):
                # Only include papers with abstracts (minimum 100 chars)
                if item.get("abstract") and len(item.get("abstract", "")) > 100:
                    papers.append(Paper(
                        title=item.get("title", ""),
                        abstract=item.get("abstract", ""),
                        year=item.get("year", 0),
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        citation_count=item.get("citationCount", 0),
                        url=item.get("url", ""),
                        tldr=item.get("tldr", {}).get("text") if item.get("tldr") else None,
                        paper_id=item.get("paperId", "")
                    ))

            return papers

        except requests.exceptions.RequestException as e:
            print(f"  Network error during search: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using local model (with caching)"""
        # Create cache key from text
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # Check cache first
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Truncate text to reasonable length
        truncated_text = text[:1000]

        # Use sentence-transformers model if available
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(truncated_text, convert_to_numpy=True)
                # Convert numpy array to list for JSON serialization
                embedding_list = embedding.tolist()
                self._embedding_cache[cache_key] = embedding_list
                return embedding_list
            except Exception as e:
                print(f"  Error generating embedding: {e}")
                return self._simple_embedding(truncated_text)
        else:
            # Fallback to simple embedding
            return self._simple_embedding(truncated_text)

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding when model unavailable (basic but functional)"""
        import numpy as np

        words = text.lower().split()
        # Create a basic frequency vector (384 dimensions to match MiniLM output size)
        vec = np.zeros(384)
        for i, word in enumerate(words[:384]):
            vec[i] = hash(word) % 100 / 100.0

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently"""
        embeddings = []

        if self.embedding_model:
            # Batch processing with sentence-transformers (faster)
            try:
                truncated_texts = [text[:1000] for text in texts]
                batch_embeddings = self.embedding_model.encode(
                    truncated_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings = [emb.tolist() for emb in batch_embeddings]
            except Exception as e:
                print(f" Batch embedding error: {e}, falling back to individual processing")
                # Fallback to individual processing
                for text in texts:
                    embeddings.append(self.get_embedding(text))
        else:
            # Process individually with simple embeddings
            for text in texts:
                embeddings.append(self.get_embedding(text))

        return embeddings