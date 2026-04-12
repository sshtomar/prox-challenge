"""
Vulcan OmniPro 220 Multimodal Support Agent
Built on the Claude Agent SDK.

Usage:
    python agent.py "What's the duty cycle for MIG at 200A on 240V?"
    python agent.py  # interactive mode
"""

import asyncio
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    query,
    tool,
    create_sdk_mcp_server,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
KB = BASE / "knowledge_base"
PAGES_DIR = KB / "pages"
DESCRIPTIONS_DIR = KB / "page_descriptions"
MARKDOWN_DIR = KB / "markdown"

# ---------------------------------------------------------------------------
# Load knowledge base into memory (small enough to keep resident)
# ---------------------------------------------------------------------------
with open(KB / "specs.json") as f:
    SPECS = json.load(f)

with open(KB / "troubleshooting.json") as f:
    TROUBLESHOOTING = json.load(f)

with open(KB / "page_index.json") as f:
    PAGE_INDEX = json.load(f)

with open(KB / "pages_manifest.json") as f:
    PAGES_MANIFEST = json.load(f)

# Pre-load markdown text
MANUAL_TEXT = (MARKDOWN_DIR / "owner-manual.md").read_text()

# Pre-load all page descriptions into a dict keyed by filename stem
PAGE_DESCRIPTIONS: dict[str, str] = {}
for md_file in sorted(DESCRIPTIONS_DIR.glob("*.md")):
    PAGE_DESCRIPTIONS[md_file.stem] = md_file.read_text()

# ---------------------------------------------------------------------------
# BM25 Index -- chunk manual by sections for relevance-ranked search
# ---------------------------------------------------------------------------

def _build_bm25_index(text: str) -> tuple[BM25Okapi, list[str]]:
    """Split manual into section chunks and build a BM25 index."""
    # Split on markdown headings (## lines)
    raw_chunks = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    # Filter out tiny fragments (< 20 chars) and keep non-empty
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= 20]

    # Tokenize: lowercase, split on whitespace/punctuation
    tokenized = [re.findall(r"[a-z0-9]+", chunk.lower()) for chunk in chunks]
    index = BM25Okapi(tokenized)
    return index, chunks

BM25_INDEX, MANUAL_CHUNKS = _build_bm25_index(MANUAL_TEXT)

# ---------------------------------------------------------------------------
# Custom Tools
# ---------------------------------------------------------------------------


@tool(
    "lookup_specs",
    "Look up exact specifications for the Vulcan OmniPro 220 welder. "
    "Use this for any question about duty cycles, amperage ranges, voltage, "
    "wire sizes, weldable materials, power requirements, polarity configurations, "
    "feed roller settings, or contact tip sizes. Returns precise values from "
    "the manufacturer's spec sheet -- never estimate these values.",
    {"query_type": str},
)
async def lookup_specs(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(SPECS, indent=2)}]
    }


@tool(
    "lookup_troubleshooting",
    "Look up troubleshooting information for the Vulcan OmniPro 220. "
    "Use this when the user describes a problem (porosity, arc not stable, "
    "wire not feeding, welder won't start, LCD not lighting, weak arc, etc.). "
    "Returns structured problem -> causes -> solutions from the manual.",
    {"process": str},
)
async def lookup_troubleshooting(args: dict[str, Any]) -> dict[str, Any]:
    process = args.get("process", "").lower()
    if "tig" in process or "stick" in process:
        data = TROUBLESHOOTING.get("tig_stick", {})
    elif "diag" in process or "weld qual" in process or "bead" in process:
        data = TROUBLESHOOTING.get("weld_diagnosis", {})
    else:
        # Return everything for MIG/Flux or general queries
        data = TROUBLESHOOTING
    return {
        "content": [{"type": "text", "text": json.dumps(data, indent=2)}]
    }


@tool(
    "get_page_description",
    "Get a detailed text description of a specific page from the manual. "
    "Use this to understand what is shown on a visual page (diagrams, photos, "
    "schematics, LCD screenshots) without needing the actual image. "
    "Available pages with descriptions: 7 (specs), 8 (front panel), 9 (interior), "
    "12 (feed roller), 13 (DCEN polarity), 14 (DCEP polarity + gas), "
    "17 (MIG LCD screens), 19 (MIG duty cycles), 24 (TIG cable setup), "
    "26 (electrode grinding + TIG torch assembly), 27 (Stick cable setup), "
    "29 (TIG/Stick duty cycles), 30 (TIG LCD screens), 32 (Stick LCD screens), "
    "34 (strike test), 35 (weld diagnosis photos), 41 (maintenance), "
    "45 (wiring schematic), 47 (assembly diagram). "
    "Also: quick-start-guide pages 1-2, selection-chart page 1.",
    {"document": str, "page_number": int},
)
async def get_page_description(args: dict[str, Any]) -> dict[str, Any]:
    doc = args.get("document", "owner-manual")
    page = args.get("page_number", 1)

    # Normalize document name
    if "quick" in doc.lower() or "start" in doc.lower():
        key = f"quick-start-guide_p{page:03d}"
    elif "select" in doc.lower() or "chart" in doc.lower():
        key = f"selection-chart_p{page:03d}"
    else:
        key = f"owner-manual_p{page:03d}"

    desc = PAGE_DESCRIPTIONS.get(key)
    if desc:
        return {"content": [{"type": "text", "text": desc}]}
    return {
        "content": [
            {
                "type": "text",
                "text": f"No detailed description available for {key}. "
                f"Available descriptions: {', '.join(sorted(PAGE_DESCRIPTIONS.keys()))}",
            }
        ],
        "isError": True,
    }


@tool(
    "get_page_image",
    "Get the actual page image from the manual as a base64-encoded PNG. "
    "Use this when the user needs to SEE a diagram, schematic, photo, or "
    "visual instruction -- not just read a text description of it. "
    "This is essential for polarity setup diagrams, weld diagnosis photos, "
    "control panel layouts, cable routing, and exploded assembly views. "
    "Pass the result directly in your response so the user can see it.",
    {"document": str, "page_number": int},
)
async def get_page_image(args: dict[str, Any]) -> dict[str, Any]:
    doc = args.get("document", "owner-manual")
    page = args.get("page_number", 1)

    if "quick" in doc.lower() or "start" in doc.lower():
        filename = f"quick-start-guide_p{page:03d}.png"
    elif "select" in doc.lower() or "chart" in doc.lower():
        filename = f"selection-chart_p{page:03d}.png"
    else:
        filename = f"owner-manual_p{page:03d}.png"

    image_path = PAGES_DIR / filename
    if not image_path.exists():
        return {
            "content": [{"type": "text", "text": f"Image not found: {filename}"}],
            "isError": True,
        }

    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "content": [
            {
                "type": "image",
                "data": image_data,
                "mimeType": "image/png",
            },
            {
                "type": "text",
                "text": f"[Page image: {filename}]",
            },
        ]
    }


@tool(
    "search_manual_text",
    "Search the full text of the owner's manual using relevance-ranked retrieval. "
    "Use this for questions that don't fit neatly into specs, troubleshooting, "
    "or visual page descriptions. Returns the most relevant manual sections, "
    "ranked by relevance to your search query. Use natural language queries "
    "for best results (e.g., 'wire feed speed settings for thin steel').",
    {"search_term": str},
)
async def search_manual_text(args: dict[str, Any]) -> dict[str, Any]:
    term = args.get("search_term", "").strip()
    if not term:
        return {
            "content": [{"type": "text", "text": "No search term provided."}],
            "isError": True,
        }

    # Tokenize query the same way we tokenized the corpus
    query_tokens = re.findall(r"[a-z0-9]+", term.lower())
    if not query_tokens:
        return {
            "content": [{"type": "text", "text": f"No valid search tokens in '{term}'."}],
            "isError": True,
        }

    # Get BM25 scores for all chunks
    scores = BM25_INDEX.get_scores(query_tokens)

    # Rank and take top 5 results with score > 0
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_results = [(idx, score) for idx, score in ranked if score > 0][:5]

    if not top_results:
        return {
            "content": [
                {"type": "text", "text": f"No relevant sections found for '{term}'."}
            ]
        }

    # Format results with section headers and relevance scores
    parts = [f"Top {len(top_results)} results for '{term}':\n"]
    for rank, (chunk_idx, score) in enumerate(top_results, 1):
        chunk = MANUAL_CHUNKS[chunk_idx]
        # Truncate very long sections to keep context manageable
        if len(chunk) > 1500:
            chunk = chunk[:1500] + "\n... [section truncated]"
        parts.append(f"--- Result {rank} (relevance: {score:.1f}) ---\n{chunk}")

    result = "\n\n".join(parts)
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "search_pages_by_topic",
    "Search the page index to find which pages cover a specific topic. "
    "Use this to find the right page number before fetching a description "
    "or image. Returns page numbers, sections, and content types.",
    {"topic": str},
)
async def search_pages_by_topic(args: dict[str, Any]) -> dict[str, Any]:
    topic = args.get("topic", "").lower()
    results = []

    for doc_name, doc_data in PAGE_INDEX.items():
        for page_info in doc_data.get("pages", []):
            topics_str = " ".join(page_info.get("topics", [])).lower()
            section = page_info.get("section", "").lower()
            visual = (page_info.get("visual_content") or "").lower()

            if topic in topics_str or topic in section or topic in visual:
                results.append(
                    {
                        "document": doc_name,
                        "page": page_info["page"],
                        "file": page_info["file"],
                        "section": page_info.get("section"),
                        "content_type": page_info.get("content_type"),
                        "topics": page_info.get("topics"),
                        "visual_content": page_info.get("visual_content"),
                        "critical_data": page_info.get("critical_data", False),
                    }
                )

    if not results:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"No pages found for topic '{topic}'. "
                    "Try broader terms like: polarity, duty cycle, troubleshooting, "
                    "TIG, MIG, stick, wire, safety, maintenance, parts, schematic.",
                }
            ]
        }

    return {
        "content": [{"type": "text", "text": json.dumps(results, indent=2)}]
    }


@tool(
    "get_polarity_quick_reference",
    "Get the polarity configuration for a specific welding process. "
    "Use this whenever someone asks about cable setup, which socket to use, "
    "ground clamp placement, or polarity for any process (MIG, Flux-Cored, TIG, Stick).",
    {"process": str},
)
async def get_polarity_quick_reference(args: dict[str, Any]) -> dict[str, Any]:
    configs = SPECS.get("polarity_configurations", {})
    process = args.get("process", "").lower()

    if "flux" in process or "fcaw" in process:
        key = "Flux-Cored_gasless"
    elif "mig" in process or "gmaw" in process or "solid" in process:
        key = "MIG_solid_core"
    elif "tig" in process or "gtaw" in process:
        key = "TIG"
    elif "stick" in process or "smaw" in process:
        key = "Stick"
    else:
        # Return all configurations
        return {
            "content": [{"type": "text", "text": json.dumps(configs, indent=2)}]
        }

    config = configs.get(key, {})
    if config:
        return {"content": [{"type": "text", "text": json.dumps(config, indent=2)}]}
    return {
        "content": [{"type": "text", "text": f"No polarity config found for '{process}'."}],
        "isError": True,
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

welding_server = create_sdk_mcp_server(
    name="welding",
    version="1.0.0",
    tools=[
        lookup_specs,
        lookup_troubleshooting,
        get_page_description,
        get_page_image,
        search_manual_text,
        search_pages_by_topic,
        get_polarity_quick_reference,
    ],
)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Vulcan OmniPro 220 Technical Support Agent.

You help users set up, operate, troubleshoot, and maintain their Vulcan OmniPro 220 multiprocess welding system (Item #57812). This welder supports MIG, Flux-Cored, TIG, and Stick welding on both 120V and 240V input.

## Your Tools

You have access to the complete owner's manual, quick-start guide, and process selection chart through these tools:

- **lookup_specs** -- Get exact specifications (duty cycles, amperage, wire sizes, materials). ALWAYS use this for numerical values. Never estimate or recall specs from memory.
- **lookup_troubleshooting** -- Get structured problem/cause/solution data when a user describes an issue.
- **get_polarity_quick_reference** -- Get cable setup and polarity for any welding process. Use whenever someone asks about cable connections or socket assignments.
- **get_page_description** -- Get a detailed text description of any visual page (diagrams, schematics, LCD screens).
- **get_page_image** -- Get the actual page image as PNG. Use when the user needs to SEE something visual.
- **search_manual_text** -- Relevance-ranked search across the manual. Use natural language queries for best results.
- **search_pages_by_topic** -- Find which pages cover a topic.

## Query Routing

Use this guide to pick the right tool first and reduce unnecessary calls:

- **Spec/number questions** (duty cycle, amperage, wire size, voltage) -> lookup_specs
- **Problem/symptom descriptions** (arc unstable, wire not feeding, welder won't start) -> lookup_troubleshooting
- **Cable/socket/polarity questions** -> get_polarity_quick_reference, then get_page_image for the diagram
- **"How do I set up..." questions** -> search_manual_text for the procedure, then get_page_image if visual
- **Visual/diagram requests** -> search_pages_by_topic to find the right page, then get_page_image
- **General "where in the manual..." questions** -> search_pages_by_topic

## Rules

1. **Never guess technical values.** Always use lookup_specs for duty cycles, amperage, wire sizes, gas flow rates, or any numerical specification. If a value is not in your tools, explicitly say: "I don't have that information in the manual" -- never estimate or infer a number.
2. **Show diagrams when they help.** If a user asks about polarity setup, cable routing, controls layout, or weld diagnosis, fetch the relevant page image so they can see it. A picture of which cable goes in which socket is worth more than a paragraph of text.
3. **Safety first.** Always include relevant safety warnings. Bold them. If someone is about to do something the manual warns against, flag it clearly.
4. **Cross-reference when needed.** Many questions require combining information from multiple sections (e.g., troubleshooting + polarity + specs). Use multiple tools in a single response.
5. **Ask clarifying questions.** If a question is ambiguous (e.g., "what polarity do I need?" without specifying the process), ask which welding process they're using.
6. **Be practical.** Your user is standing in their garage trying to get this welder working. Be direct, actionable, and clear. Skip jargon unless it's what the manual uses.
7. **Cite your sources.** When referencing specifications, procedures, or troubleshooting steps, cite the source (e.g., "per the specs sheet", "see manual p.13", "from the troubleshooting guide"). This helps users verify information and find it in their own manual.
8. **Admit limits honestly.** If a question is about a different product, aftermarket parts, or topics not covered in the Vulcan OmniPro 220 manual, say so clearly. Do not fabricate answers about competitor products, third-party accessories, or information outside your knowledge base.

## Tone

Helpful, knowledgeable, patient. Like an experienced welder friend who actually read the manual. Not condescending, not overly formal.
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_interactive():
    """Run the agent in interactive multi-turn mode."""
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=10,
        max_budget_usd=0.05,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"welding": welding_server},
        allowed_tools=["mcp__welding__*"],
        permission_mode="acceptEdits",
        cwd=str(BASE),
    )

    print("Vulcan OmniPro 220 Support Agent")
    print("Type your question, or 'quit' to exit.\n")

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            await client.query(user_input)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\nAgent: {block.text}")
                elif isinstance(message, ResultMessage):
                    if message.subtype == "error":
                        print(f"\n[Error: {message.result}]")
            print()


async def run_single(prompt: str):
    """Run a single query and print the result."""
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=10,
        max_budget_usd=0.05,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"welding": welding_server},
        allowed_tools=["mcp__welding__*"],
        permission_mode="acceptEdits",
        cwd=str(BASE),
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        elif isinstance(message, ResultMessage):
            if message.subtype == "error":
                print(f"[Error: {message.result}]")


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(run_single(prompt))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
