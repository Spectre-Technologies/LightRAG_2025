import os
import asyncio
import json
from typing import List, Dict, Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------
# ⚠️  Set these two values to match the cruise line you want to test.
#     1. LINE_NAME   – Friendly display name used inside the analysis report.
#     2. GRAPH_DIR   – Path to the existing LightRAG knowledge-base directory.
# ---------------------------------------------------------------------------
LINE_NAME: str = "MSC Cruises"  # ⇦ CHANGE ME
GRAPH_DIR: str = "./msc_cruises_graph/msc_cruises"      # ⇦ CHANGE ME

# Hardcode the OpenAI API key (WARNING: for demo/testing only, do not commit secrets)
 
# ---------------------------------------------------------------------------
# Global configuration (tweak only if you know what you are doing)
# ---------------------------------------------------------------------------
setup_logger("lightrag", level="INFO")  # Change to DEBUG for verbose output

# Search mode to use for every query (hybrid | mix | naive)
SEARCH_MODE = "hybrid"

# How many source snippets to retrieve for each answer (optional)
MAX_SOURCES = 3

# Representative customer-service queries a travel-agency chatbot should handle
TEST_QUERIES: List[str] = [
    "What is the refund policy if I cancel my cruise?",
    "Are there any special discounts for military personnel when booking a cruise?",
    "Which travel documents do I need to board an international cruise?",
    "Can I bring my own alcohol on board and what are the restrictions?",
    "How can I update my dining preference after booking my cruise?",
    "What are the current COVID-19 vaccination or testing requirements for embarkation?",
    "Does the cruise line offer internet packages and how much do they cost?",
    "Is travel insurance available and what does it cover?",
    "Can I change my cruise itinerary or sailing date after booking?",
    "What are the age requirements for booking and travelling alone on a cruise?",
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

async def init_rag(working_dir: str) -> LightRAG:
    """Initialise a LightRAG instance for the given working directory."""
    if not os.path.exists(working_dir):
        raise FileNotFoundError(
            f"Knowledge-base directory '{working_dir}' does not exist. "
            "Please run the ingestion pipeline first."
        )

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=500,
        chunk_overlap_token_size=100,
    )
    await rag.initialize_storages()
    return rag


async def query_rag(
    rag: LightRAG,
    query: str,
    mode: str = SEARCH_MODE,
    max_sources: int = MAX_SOURCES,
) -> Dict[str, Any]:
    """Run a single query against a LightRAG instance and gather answer + sources."""
    param = QueryParam(mode=mode)
    answer = await rag.aquery(query, param=param)

    # Try to fetch top-k sources for traceability (if available)
    sources = []
    if hasattr(rag, "aget_sources"):
        raw_sources = await rag.aget_sources(query, param=param)
        for src in raw_sources[:max_sources]:
            sources.append(
                {
                    "file": os.path.basename(src.metadata.get("file_path", "?")),
                    "score": src.score,
                    "preview": src.content[:300]
                    + ("..." if len(src.content) > 300 else ""),
                }
            )

    return {"answer": answer, "sources": sources}


async def run_all_queries(rag: LightRAG) -> List[Dict[str, Any]]:
    """Execute all TEST_QUERIES against the provided RAG instance in parallel."""
    tasks = [query_rag(rag, q) for q in TEST_QUERIES]
    return await asyncio.gather(*tasks)


async def generate_analysis(
    results: List[Dict[str, Any]],
    line_name: str,
) -> str:
    """Send the results to GPT-4o for a qualitative assessment."""
    analysis_prompt = (
        "You are an expert knowledge-base evaluator for a cruise travel agency.\n"
        f"The following answers were generated using the knowledge graph for '{line_name}'.\n\n"
        "For each customer-service query below, please:\n"
        "1. Evaluate the accuracy, completeness and customer-friendliness of the answer.\n"
        "2. Point out any factual discrepancies, missing details, or hallucinations.\n"
        "3. Suggest concrete improvements for future knowledge-base ingestion.\n\n"
        "Use bullet-points and clear headings.\n\n"
        f"Here is the data in JSON format:\n{json.dumps(list(zip(TEST_QUERIES, results)), indent=2)}"
    )

    analysis = await gpt_4o_mini_complete(
        analysis_prompt,
        system_prompt="You are a meticulous analyst tasked with evaluating a single knowledge base.",
        max_tokens=1024,
    )
    return analysis


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"Initialising knowledge graph for '{LINE_NAME}' …")
    rag = await init_rag(GRAPH_DIR)

    print("Running queries against the graph …")
    results = await run_all_queries(rag)

    print("\n=== RAW RESULTS COLLECTED ===\n")
    for idx, query in enumerate(TEST_QUERIES):
        print(f"Query {idx + 1}: {query}")
        print("—" * 80)
        print("Answer:\n", results[idx]["answer"], "\n")
        print("Sources:")
        for src in results[idx]["sources"]:
            print(
                f"  • {src['file']} (score: {src['score']:.4f})\n    → {src['preview']}"
            )
        print("=" * 80)

    print("\nGenerating analytical report via GPT-4o …\n")
    analysis_report = await generate_analysis(results, LINE_NAME)

    print("\n=== ANALYSIS REPORT ===\n")
    print(analysis_report)

    # Write the report to a markdown file
    report_filename = f"{LINE_NAME.lower().replace(' ', '_')}_graph_report.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"# {LINE_NAME} Knowledge Graph Report\n\n")
        f.write(analysis_report)

    print(f"\nAnalysis report written to {report_filename}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.") 