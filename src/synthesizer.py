import anthropic
from typing import List, Dict, Optional
from src.models import Paper

class ClaudeSynthesizer:
    def __init__(self, claude_api_key: str):
        self.claude_client = (
            anthropic.Anthropic(api_key=claude_api_key))

    def generate_search_queries(self, user_question: str) -> List[str]:
        """Use Claude to generate multiple search queries"""
        prompt = f"""Given this medical/scientific question, generate 3 different search queries 
that would help find relevant academic papers. Make them specific and use scientific terminology.

Question: {user_question}

Return only the queries, one per line."""

        message = self.claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        queries = [q.strip() for q in message.content[0].text.strip().split('\n') if q.strip()]
        return queries[:3]

    def detect_contradictions(self, papers: List[Paper]) -> Dict:
        """Use Claude to detect contradictions between papers"""
        if len(papers) < 2:
            return {"analysis": "Not enough papers to detect contradictions."}

        # Prepare paper summaries for comparison
        paper_summaries = []
        for i, paper in enumerate(papers[:6], 1):  # Compare top 6 papers
            summary = f"Paper {i} ({paper.year}, {paper.citation_count} citations):\n{paper.abstract[:600]}"
            paper_summaries.append(summary)

        prompt = f"""Analyze these research papers and identify any contradictions or conflicting findings.

{chr(10).join(paper_summaries)}

List any clear contradictions you find. For each contradiction:
1. State which papers conflict (by number)
2. Describe the nature of the disagreement
3. Suggest possible reasons (methodology, sample size, etc.)

If there are no significant contradictions, say "No major contradictions detected."
"""

        message = self.claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"analysis": message.content[0].text}

    def synthesize_findings(self, user_question: str, papers: List[Paper],
                            contradictions: Optional[Dict] = None) -> str:
        """Use Claude to synthesize findings"""

        paper_summaries = []
        for i, paper in enumerate(papers, 1):
            summary = f"""
[{i}] {paper.title} ({paper.year})
    Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
    Citations: {paper.citation_count} | Relevance: {paper.relevance_score:.2f}
    {'TLDR: ' + paper.tldr if paper.tldr else ''}
    Abstract: {paper.abstract[:700]}...
    URL: {paper.url}
"""
            paper_summaries.append(summary)

        papers_text = "\n".join(paper_summaries)

        contradiction_section = ""
        if contradictions:
            contradiction_section = f"\n\nCONTRADICTION ANALYSIS:\n{contradictions['analysis']}\n"

        prompt = f"""You are a medical research assistant helping users understand current scientific evidence.

USER QUESTION: "{user_question}"

RELEVANT RESEARCH PAPERS:
{papers_text}
{contradiction_section}

Provide a comprehensive answer that:
1. **Direct Answer**: Start with a clear, direct response to the question
2. **Key Findings**: Summarize the main consensus from the research
3. **Evidence Quality**: Note the strength of evidence (citation counts, recency, sample sizes)
4. **Nuances & Limitations**: Discuss any caveats, contradictions, or areas of uncertainty
5. **Clinical Relevance**: If applicable, mention practical implications

Use citations [1], [2], etc. Write in clear, accessible language while maintaining scientific rigor.
Structure with short headers. Keep it focused and actionable."""

        message = self.claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text