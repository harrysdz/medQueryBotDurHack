from src.bot import EnhancedMedQueryBot
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    claude_key = os.environ['CLAUDE_API_KEY']
    s2_key = os.environ['SEMANTIC_SCHOLAR_API_KEY']

    if not claude_key or not s2_key:
        print("Error: API keys not found in .env file")
        exit(1)
    bot = EnhancedMedQueryBot(
        claude_api_key = claude_key,
        s2_api_key = s2_key
    )

    question = "What is the comparative efficacy and safety of SGLT2 inhibitors versus GLP-1 receptor agonists in reducing cardiovascular events in patients with type 2 diabetes?"

    result = bot.answer_question(
        question,
        papers_per_query=15,
        detect_contradictions=True,
        top_k_papers=8
    )

    print("="*80)
    print("ANSWER:")
    print("="*80)
    print(result["answer"])
    print("\n" + "="*80)
    print(f"📊 Analysis based on {result['papers_used']} of {result['papers_analyzed']} papers found")
    print("="*80)

    print("\n📑 Top Papers Used:")
    for i, paper in enumerate(result["papers"][:5], 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Year: {paper.year} | Citations: {paper.citation_count} | Score: {paper.relevance_score:.3f}")
        print(f"   {paper.url}")