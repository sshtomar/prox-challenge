"""
FastAPI server wrapping the Vulcan OmniPro 220 agent.
Serves the frontend and streams agent responses via SSE.
Instrumented with Pydantic Logfire for observability.
"""

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import logfire

logfire.configure(
    service_name="prox-welding-agent",
    service_version="1.0.0",
    environment="development",
    send_to_logfire="if-token-present",
)

from rank_bm25 import BM25Okapi

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolAnnotations,
    ToolUseBlock,
    tool,
    create_sdk_mcp_server,
)

# Note: logfire.instrument_claude_agent_sdk() not available in current version.
# Using manual spans on tools and chat endpoint instead.

BASE = Path(__file__).parent
KB = BASE / "knowledge_base"
PAGES_DIR = KB / "pages"
DESCRIPTIONS_DIR = KB / "page_descriptions"
MARKDOWN_DIR = KB / "markdown"

# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------
with open(KB / "specs.json") as f:
    SPECS = json.load(f)

with open(KB / "troubleshooting.json") as f:
    TROUBLESHOOTING = json.load(f)

with open(KB / "page_index.json") as f:
    PAGE_INDEX = json.load(f)

MANUAL_TEXT = (MARKDOWN_DIR / "owner-manual.md").read_text()

PAGE_DESCRIPTIONS: dict[str, str] = {}
for md_file in sorted(DESCRIPTIONS_DIR.glob("*.md")):
    PAGE_DESCRIPTIONS[md_file.stem] = md_file.read_text()


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

def _build_bm25_index(text: str) -> tuple[BM25Okapi, list[str]]:
    """Split manual into section chunks and build a BM25 index."""
    raw_chunks = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= 20]
    tokenized = [re.findall(r"[a-z0-9]+", chunk.lower()) for chunk in chunks]
    index = BM25Okapi(tokenized)
    return index, chunks

BM25_INDEX, MANUAL_CHUNKS = _build_bm25_index(MANUAL_TEXT)


# ---------------------------------------------------------------------------
# Session Management (ClaudeSDKClient per session for native multi-turn)
# ---------------------------------------------------------------------------

@dataclass
class Session:
    client: ClaudeSDKClient
    lock: asyncio.Lock
    last_active: float


_sessions: dict[str, Session] = {}
_SESSION_TTL = 3600  # 1 hour


async def _get_or_create_session(session_id: str) -> Session:
    """Get existing session or create a new ClaudeSDKClient."""
    await _cleanup_sessions()
    if session_id in _sessions:
        session = _sessions[session_id]
        session.last_active = time.time()
        return session

    options = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        effort="low",
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"welding": welding_server},
        allowed_tools=["mcp__welding__*"],
        permission_mode="acceptEdits",
        max_turns=8,
        include_partial_messages=True,
        cwd=str(BASE),
    )
    client = ClaudeSDKClient(options=options)
    await client.__aenter__()
    session = Session(
        client=client,
        lock=asyncio.Lock(),
        last_active=time.time(),
    )
    _sessions[session_id] = session
    return session


async def _destroy_session(session_id: str) -> None:
    """Close and remove a session."""
    session = _sessions.pop(session_id, None)
    if session:
        try:
            await session.client.__aexit__(None, None, None)
        except Exception:
            pass


async def _cleanup_sessions() -> None:
    """Remove sessions idle longer than TTL."""
    cutoff = time.time() - _SESSION_TTL
    expired = [sid for sid, s in _sessions.items() if s.last_active < cutoff]
    for sid in expired:
        await _destroy_session(sid)


# ---------------------------------------------------------------------------
# Tools (same as agent.py but defined here to avoid import issues)
# ---------------------------------------------------------------------------

@tool(
    "lookup_specs",
    "Look up exact specifications for the Vulcan OmniPro 220 welder. "
    "Pass a process name (MIG, Flux-Cored, TIG, Stick) to get targeted specs, "
    "or 'all' for everything. Covers duty cycles, amperage, voltage, "
    "wire sizes, polarity, feed rollers, and contact tips.",
    {"query_type": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def lookup_specs(args: dict[str, Any]) -> dict[str, Any]:
    qt = args.get("query_type", "").lower()
    with logfire.span("tool_lookup_specs", query_type=qt):
        # Return targeted subset when a process is specified
        process_map = {
            "mig": ("MIG", "MIG_solid_core"),
            "gmaw": ("MIG", "MIG_solid_core"),
            "flux": ("MIG", "Flux-Cored_gasless"),
            "fcaw": ("MIG", "Flux-Cored_gasless"),
            "tig": ("TIG", "TIG"),
            "gtaw": ("TIG", "TIG"),
            "stick": ("Stick", "Stick"),
            "smaw": ("Stick", "Stick"),
        }
        for keyword, (spec_key, pol_key) in process_map.items():
            if keyword in qt:
                result = {
                    "specifications": {spec_key: SPECS["specifications"][spec_key]},
                    "polarity_configurations": {pol_key: SPECS["polarity_configurations"][pol_key]},
                    "duty_cycle_explanation": SPECS["duty_cycle_explanation"],
                }
                if spec_key == "MIG":
                    result["wire_specifications"] = SPECS["wire_specifications"]
                    result["feed_roller_configuration"] = SPECS["feed_roller_configuration"]
                    result["contact_tip_sizes"] = SPECS["contact_tip_sizes"]
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        return {"content": [{"type": "text", "text": json.dumps(SPECS, indent=2)}]}


@tool(
    "lookup_troubleshooting",
    "Look up troubleshooting information. Use when the user describes a problem. "
    "Pass 'mig', 'flux', 'tig', 'stick', or 'weld diagnosis' to get the right section.",
    {"process": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def lookup_troubleshooting(args: dict[str, Any]) -> dict[str, Any]:
    process = args.get("process", "").lower()
    with logfire.span("tool_lookup_troubleshooting", process=process):
        if "diag" in process or "bead" in process or "penetration" in process:
            data = TROUBLESHOOTING.get("weld_diagnosis", {})
        elif "tig" in process or "stick" in process:
            data = TROUBLESHOOTING.get("tig_stick", {})
        else:
            data = TROUBLESHOOTING.get("mig_flux_cored", {})
        return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]}


@tool(
    "get_page_description",
    "Get a detailed text description of a specific page from the manual.",
    {"document": str, "page_number": int},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_page_description(args: dict[str, Any]) -> dict[str, Any]:
    doc = args.get("document", "owner-manual")
    page = args.get("page_number", 1)
    with logfire.span("tool_get_page_description", document=doc, page_number=page):
        if "quick" in doc.lower():
            key = f"quick-start-guide_p{page:03d}"
        elif "select" in doc.lower():
            key = f"selection-chart_p{page:03d}"
        else:
            key = f"owner-manual_p{page:03d}"
        desc = PAGE_DESCRIPTIONS.get(key)
        if desc:
            return {"content": [{"type": "text", "text": desc}]}
        return {"content": [{"type": "text", "text": f"No description for {key}."}], "isError": True}


@tool(
    "get_page_image",
    "Get the actual page image from the manual as base64 PNG.",
    {"document": str, "page_number": int},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_page_image(args: dict[str, Any]) -> dict[str, Any]:
    doc = args.get("document", "owner-manual")
    page = args.get("page_number", 1)
    with logfire.span("tool_get_page_image", document=doc, page_number=page):
        if "quick" in doc.lower():
            filename = f"quick-start-guide_p{page:03d}.png"
        elif "select" in doc.lower():
            filename = f"selection-chart_p{page:03d}.png"
        else:
            filename = f"owner-manual_p{page:03d}.png"
        image_path = PAGES_DIR / filename
        if not image_path.exists():
            return {"content": [{"type": "text", "text": f"Image not found: {filename}"}], "isError": True}
        return {"content": [
            {"type": "text", "text": f"Image is available at URL: /pages/{filename}\nEmbed it in your response using markdown: ![{filename}](/pages/{filename})"},
        ]}


@tool(
    "search_manual_text",
    "Search the full text of the owner's manual using relevance-ranked retrieval. "
    "Returns the most relevant manual sections ranked by relevance. "
    "Use natural language queries for best results.",
    {"search_term": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def search_manual_text(args: dict[str, Any]) -> dict[str, Any]:
    term = args.get("search_term", "").strip()
    with logfire.span("tool_search_manual_text", search_term=term):
        if not term:
            return {"content": [{"type": "text", "text": "No search term."}], "isError": True}

        query_tokens = re.findall(r"[a-z0-9]+", term.lower())
        if not query_tokens:
            return {"content": [{"type": "text", "text": f"No valid tokens in '{term}'."}], "isError": True}

        scores = BM25_INDEX.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_results = [(idx, score) for idx, score in ranked if score > 0][:5]

        logfire.info(
            "BM25 search completed",
            search_term=term,
            results_found=len(top_results),
            top_score=round(top_results[0][1], 2) if top_results else 0,
        )

        if not top_results:
            return {"content": [{"type": "text", "text": f"No relevant sections for '{term}'."}]}

        parts = [f"Top {len(top_results)} results for '{term}':\n"]
        for rank, (chunk_idx, score) in enumerate(top_results, 1):
            chunk = MANUAL_CHUNKS[chunk_idx]
            if len(chunk) > 1500:
                chunk = chunk[:1500] + "\n... [section truncated]"
            parts.append(f"--- Result {rank} (relevance: {score:.1f}) ---\n{chunk}")

        return {"content": [{"type": "text", "text": "\n\n".join(parts)}]}


@tool(
    "search_pages_by_topic",
    "Search the page index to find which pages cover a specific topic.",
    {"topic": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def search_pages_by_topic(args: dict[str, Any]) -> dict[str, Any]:
    topic = args.get("topic", "").lower()
    with logfire.span("tool_search_pages_by_topic", topic=topic):
        results = []
        for doc_name, doc_data in PAGE_INDEX.items():
            for p in doc_data.get("pages", []):
                topics_str = " ".join(p.get("topics", [])).lower()
                if topic in topics_str or topic in p.get("section", "").lower():
                    results.append({"document": doc_name, "page": p["page"],
                                    "section": p.get("section"), "topics": p.get("topics")})
        if not results:
            return {"content": [{"type": "text", "text": f"No pages for '{topic}'."}]}
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}


@tool(
    "get_polarity_quick_reference",
    "Get polarity configuration for a welding process.",
    {"process": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_polarity_quick_reference(args: dict[str, Any]) -> dict[str, Any]:
    configs = SPECS.get("polarity_configurations", {})
    process = args.get("process", "").lower()
    with logfire.span("tool_get_polarity", process=process):
        if "flux" in process:
            key = "Flux-Cored_gasless"
        elif "mig" in process or "solid" in process:
            key = "MIG_solid_core"
        elif "tig" in process:
            key = "TIG"
        elif "stick" in process:
            key = "Stick"
        else:
            return {"content": [{"type": "text", "text": json.dumps(configs, indent=2)}]}
        config = configs.get(key, {})
        return {"content": [{"type": "text", "text": json.dumps(config, indent=2)}]}


# ---------------------------------------------------------------------------
# MCP Server + System Prompt
# ---------------------------------------------------------------------------

welding_server = create_sdk_mcp_server(
    name="welding",
    version="1.0.0",
    tools=[
        lookup_specs, lookup_troubleshooting, get_page_description,
        get_page_image, search_manual_text, search_pages_by_topic,
        get_polarity_quick_reference,
    ],
)

SYSTEM_PROMPT = """You are the Vulcan OmniPro 220 Technical Support Agent, built by Prox.

You help users set up, operate, troubleshoot, and maintain their Vulcan OmniPro 220 multiprocess welding system (Item #57812). This welder supports MIG, Flux-Cored, TIG, and Stick welding on both 120V and 240V input.

## Your Tools

- **lookup_specs** -- Exact specs (duty cycles, amperage, wire sizes). ALWAYS use for numerical values.
- **lookup_troubleshooting** -- Structured problem/cause/solution data.
- **get_polarity_quick_reference** -- Cable setup and polarity per process.
- **get_page_description** -- Detailed text description of visual manual pages.
- **get_page_image** -- Actual page image as PNG.
- **search_manual_text** -- Relevance-ranked search across the manual. Use natural language queries.
- **search_pages_by_topic** -- Find pages by topic.

## Query Routing

Pick the right tool first to minimize unnecessary calls:

- **Spec/number questions** (duty cycle, amperage, wire size, voltage) -> lookup_specs
- **Problem/symptom descriptions** (arc unstable, wire not feeding, won't start) -> lookup_troubleshooting
- **Cable/socket/polarity questions** -> get_polarity_quick_reference, then get_page_image for the diagram
- **"How do I..." setup questions** -> search_manual_text for procedure, then get_page_image if visual
- **Visual/diagram requests** -> search_pages_by_topic to find the page, then get_page_image
- **General "where in the manual..." questions** -> search_pages_by_topic

## Rules

1. Never guess technical values. Always use lookup_specs. If a value is not in your tools, explicitly say: "I don't have that information in the manual" -- never estimate or infer a number.
2. Show diagrams when they help (polarity, controls, weld diagnosis).
3. Safety first. Bold all safety warnings.
4. Cross-reference multiple sections when needed.
5. NEVER ASK CLARIFYING QUESTIONS unless the query is genuinely ambiguous about which welding PROCESS to use. For everything else -- parts, settings, procedures, terms -- search first and answer with what you find. If the user says "american air fitting", search the manual, find part #27, and tell them about it. Do not ask "what do you mean?" or "are you asking about X or Y?" when a search would answer the question. The user is standing in their garage -- every clarification round-trip wastes 30 seconds.
6. Be practical and direct. The user is in their garage trying to get this working.
7. Keep responses SHORT. Answer the question in 2-4 sentences when possible. Use a table or list only when there are multiple values to compare. Do not add background explanations the user didn't ask for. Do not end with "let me know if you need anything else" or similar filler. Latency matters -- a fast short answer beats a slow thorough one.
8. Cite sources using ONLY the bracket format: [p.7], [p.13], [p.46], [qsg.1], [chart.1]. NEVER write bare page references like "p.42" or "(see page 13)" -- ALWAYS wrap in brackets: [p.42], [p.13]. NEVER write [parts list], [specs], [manual], or any free-text citation. The parts list is [p.46], the assembly diagram is [p.47], specs are [p.7]. For page ranges use [p.42-43].
9. Admit limits honestly. If a question is about a different product, aftermarket parts, or topics outside the Vulcan OmniPro 220 manual, say so clearly. Do not fabricate answers.

## Code Generation -- Visual Responses

The frontend renders ```html code blocks as live interactive artifacts in sandboxed iframes. Use this ONLY when the user explicitly asks for a visual, diagram, or interactive tool, OR when the answer is inherently spatial/visual (e.g., cable routing between sockets).

**Default to plain text/markdown.** Most questions -- even about duty cycles, specs, or troubleshooting -- are best answered with a concise text response, a table, or a numbered list. Artifacts add latency and are not needed for factual answers.

**Generate an artifact ONLY when:**
- The user explicitly asks: "show me a diagram", "draw the polarity setup", "make a calculator"
- The answer involves spatial relationships that text cannot convey (which cable plugs into which socket)
- The user asks for a configurator or interactive tool (e.g., "help me figure out settings for my material")

**Do NOT generate artifacts for:**
- Spec lookups (duty cycle, amperage, wire size) -- use text/tables
- Troubleshooting steps -- use numbered lists
- Any question that can be answered in a few sentences or a table
- Safety warnings (always text so they can't be missed)

**How to generate code (when appropriate):**
Write a SINGLE self-contained HTML block inside triple backticks with the html language tag. Rules:
- Include ALL CSS inline in a <style> tag. No external stylesheets.
- Include ALL JS inline in a <script> tag. No external scripts.
- Use clean SVG for diagrams. Use clear colors: red for negative/danger, green for positive/safe, blue for informational.
- Make it interactive where possible (hover states, click handlers, sliders).
- Use a clean white background, modern sans-serif font (system-ui), good spacing.
- Keep it focused -- one visualization per block, not a full app.
- The iframe is approximately 700px wide, so design for that.
- Always use accurate values from the manual data (call lookup_specs first if needed).

## Response Format and Citations

Use markdown formatting. For emphasis use **bold**. For lists use numbered steps.

**Always cite your sources using this exact format:** `[p.NUMBER]` where NUMBER is the manual page number.
Examples: `[p.7]`, `[p.13]`, `[p.42]`

For the quick-start guide, use `[qsg.1]` or `[qsg.2]`.
For the selection chart, use `[chart.1]`.

Place citations at the end of a paragraph or section, not on every line. One citation per source is enough -- do NOT repeat the same citation on every row of a table or every item in a list. For example:

Good (grouped):
- "The MIG duty cycle at 200A on 240V is 25%, with 100% continuous use at 115A [p.7]."
- A settings table followed by a single "[p.7]" at the bottom

Bad (repetitive -- NEVER do this):
- Every row in a table ending with "[p.7]"
- Every bullet point in a list ending with "[p.42]"
- The same citation appearing more than twice in one response

STRICT RULE: When multiple points come from the same page, cite the page ONCE at the end of the section. For example, a 5-bullet troubleshooting list from page 42 gets ONE [p.42] after the last bullet, not five.

You may mix text and code blocks in the same response -- e.g., a brief text explanation followed by an interactive diagram.
"""

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Prox - Vulcan OmniPro 220 Agent")
logfire.instrument_fastapi(app)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
app.mount("/pages", StaticFiles(directory=str(PAGES_DIR)), name="pages")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE / "static" / "index.html").read_text()


@app.get("/api/specs")
async def api_specs():
    return SPECS


@app.get("/api/specs/{process}")
async def api_process_specs(process: str):
    process = process.upper()
    spec = SPECS.get("specifications", {}).get(process)
    if spec:
        return spec
    return JSONResponse({"error": f"Unknown process: {process}"}, status_code=404)


@app.get("/api/troubleshooting/{process}")
async def api_troubleshooting(process: str):
    p = process.lower()
    if "tig" in p or "stick" in p:
        return TROUBLESHOOTING.get("tig_stick", {})
    return TROUBLESHOOTING.get("mig_flux_cored", {})


@app.get("/api/polarity/{process}")
async def api_polarity(process: str):
    configs = SPECS.get("polarity_configurations", {})
    p = process.lower()
    if "flux" in p:
        return configs.get("Flux-Cored_gasless", {})
    elif "mig" in p:
        return configs.get("MIG_solid_core", {})
    elif "tig" in p:
        return configs.get("TIG", {})
    elif "stick" in p:
        return configs.get("Stick", {})
    return configs


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    process_context = body.get("process", "")
    voltage = body.get("voltage", "240V")
    session_id = body.get("session_id") or str(uuid.uuid4())

    context_prefix = ""
    if process_context:
        context_prefix = f"[User has selected process: {process_context}, voltage: {voltage}] "

    full_prompt = context_prefix + user_message

    logfire.info(
        "Chat request received",
        user_message=user_message,
        process=process_context,
        voltage=voltage,
        session_id=session_id,
    )

    session = await _get_or_create_session(session_id)

    async def event_stream():
        async with session.lock:
            with logfire.span(
                "agent_chat",
                user_message=user_message,
                process=process_context,
                voltage=voltage,
                session_id=session_id,
            ) as span:
                start_time = time.monotonic()
                text_blocks = 0
                total_response_chars = 0
                tools_called: list[str] = []
                tool_call_count = 0
                streamed_deltas = False
                try:
                    await session.client.query(full_prompt)
                    async for message in session.client.receive_response():
                        if isinstance(message, StreamEvent):
                            event = message.event
                            event_type = event.get("type", "")
                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    chunk = delta.get("text", "")
                                    if chunk:
                                        streamed_deltas = True
                                        total_response_chars += len(chunk)
                                        data = json.dumps({"type": "delta", "content": chunk})
                                        yield f"data: {data}\n\n"
                        elif isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    text_blocks += 1
                                    # Only send full text if we didn't already stream deltas
                                    if not streamed_deltas:
                                        data = json.dumps({"type": "text", "content": block.text})
                                        yield f"data: {data}\n\n"
                                elif isinstance(block, ToolUseBlock):
                                    tool_call_count += 1
                                    tools_called.append(block.name)
                            # Reset for next turn (tool call may trigger another response)
                            streamed_deltas = False
                        elif isinstance(message, ResultMessage):
                            latency_ms = (time.monotonic() - start_time) * 1000

                            tool_freq: dict[str, int] = {}
                            for t in tools_called:
                                tool_freq[t] = tool_freq.get(t, 0) + 1

                            span.set_attribute("response_blocks", text_blocks)
                            span.set_attribute("response_chars", total_response_chars)
                            span.set_attribute("latency_ms", round(latency_ms, 1))
                            span.set_attribute("status", message.subtype)
                            span.set_attribute("tool_call_count", tool_call_count)
                            span.set_attribute("tools_called", tools_called)
                            span.set_attribute("tool_frequency", tool_freq)

                            logfire.info(
                                "Chat completed",
                                status=message.subtype,
                                latency_ms=round(latency_ms, 1),
                                response_blocks=text_blocks,
                                response_chars=total_response_chars,
                                tool_call_count=tool_call_count,
                                tools_called=tools_called,
                                tool_frequency=tool_freq,
                            )
                            data = json.dumps({"type": "done", "status": message.subtype})
                            yield f"data: {data}\n\n"
                except Exception as e:
                    logfire.exception("Chat request failed")
                    data = json.dumps({"type": "error", "content": str(e)})
                    yield f"data: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/reset")
async def chat_reset(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")
    if session_id:
        await _destroy_session(session_id)
    return JSONResponse({"status": "ok"})


@app.post("/api/feedback")
async def feedback(request: Request):
    body = await request.json()
    vote = body.get("vote", "")
    session_id = body.get("session_id", "")
    feedback_id = body.get("feedback_id", "")
    logfire.info(
        "User feedback received",
        vote=vote,
        session_id=session_id,
        feedback_id=feedback_id,
        _tags=["feedback"],
    )
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
