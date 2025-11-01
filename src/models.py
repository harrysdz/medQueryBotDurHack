from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class Paper:
    title: str
    abstract: str
    year: int
    authors: List[str]
    citation_count: int
    url: str
    paper_id: str
    tldr: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    relevance_score: float = 0.0
