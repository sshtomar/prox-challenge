# Vulcan OmniPro 220 Support Agent

Technical support agent for the Vulcan OmniPro 220 multiprocess welder. Built on the Claude Agent SDK with a FastAPI backend and a three-panel chat interface.

**Live:** https://prox-welding-agent-production.up.railway.app

## Demo

[![Watch the demo](docs/demo-thumbnail.jpg)](https://github.com/sshtomar/prox-challenge/raw/main/docs/prox-challenge.mp4)

## Quick Start

```bash
git clone https://github.com/sshtomar/prox-challenge.git
cd prox-challenge
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
python server.py
```

Open http://localhost:8000.

## How It Works

The agent is a RAG system with 7 custom tools registered through an MCP server, backed by structured JSON, BM25-indexed markdown, and 300 DPI page images from the 48-page owner's manual.

**Request flow:**

1. Frontend sends message + process context + voltage + session_id via POST /api/chat
2. Server prepends process/voltage context to the message
3. Server finds or creates a persistent ClaudeSDKClient for the session
4. Claude Haiku reads the system prompt, decides which tools to call
5. Tools execute (lookup_specs, search_manual_text, get_page_image, etc.)
6. Agent synthesizes tool results into a response with page citations
7. Response streams to the frontend via SSE

## Architecture

### Knowledge Base

Built from 3 source PDFs processed through a two-stage pipeline: PDF rasterization at 300 DPI (pdf2image), then markdown extraction (IBM Docling).

| File | Purpose |
|------|---------|
| `specs.json` | Structured specs for all 4 processes at 120V/240V. Single source of truth for numbers. |
| `troubleshooting.json` | Problem/Cause/Solution matrices. Three sections for direct lookup. |
| `page_index.json` | Metadata for all 51 pages: sections, topics, content types. |
| `markdown/owner-manual.md` | Full text chunked into 111 sections by `##` headers for BM25 indexing. |
| `pages/*.png` | 300 DPI rasterizations of every page. Served as static files via `/pages/`. |
| `page_descriptions/*.md` | Hand-written descriptions of 22 critical visual pages. |

### Tools

7 tools, each built for a specific query pattern. All read-only, instrumented with Logfire spans.

| Tool | What It Does |
|------|-------------|
| `lookup_specs` | Returns exact specs from specs.json. Filters by process. Never estimates -- reads curated JSON. |
| `lookup_troubleshooting` | Returns structured Problem/Cause/Solution data. Routes to the right section by keyword. |
| `get_polarity_quick_reference` | Returns polarity config (DCEP/DCEN, which socket) per process. |
| `search_manual_text` | BM25 relevance-ranked search across 111 manual sections. Top 5 results with scores. |
| `get_page_description` | Returns a text description of a visual page (diagrams, schematics, LCD screens). |
| `get_page_image` | Returns the URL to a page PNG for inline display in chat. |
| `search_pages_by_topic` | Searches page_index.json to find which pages cover a topic. |

### Anti-Hallucination Design

Hallucination in a welding manual agent is dangerous -- a wrong duty cycle could cause a fire, wrong polarity could damage equipment. Five layers of defense:

1. **Structured data store** -- specs and troubleshooting are curated JSON, not LLM-extracted text
2. **System prompt guardrails** -- "never guess technical values", "admit limits honestly"
3. **Query routing** -- spec questions go to lookup_specs, not to free-text search
4. **Citation requirements** -- bracket format [p.7] forces the agent to trace claims to sources
5. **Adversarial eval suite** -- 8 test cases (H01-H08) specifically test for hallucination. 8/8 passed.

### Frontend

Three-panel layout designed for someone standing in their garage with a phone:

- **Left (220px):** Process selector with quick setup wizard (material + thickness)
- **Center:** Chat with markdown rendering, tables, inline images, and sandboxed HTML artifacts
- **Right (280px):** Live context panel showing duty cycles, polarity, and specs for the selected process

Voice input via Web Speech API. Typing indicator with animated dots. HTML code blocks render as interactive artifacts in sandboxed iframes.

### Deployment

Single FastAPI process on Railway. Loads 33MB knowledge base into memory at startup, serves both frontend and API. SSE streaming with `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers.

| Parameter | Value |
|-----------|-------|
| Model | claude-haiku-4-5-20251001 |
| Effort | low |
| Max turns | 8 |
| Max budget | $0.25/query |
| Avg query cost | ~$0.03 |
| Avg response time | 15-20s |

### Observability

Every request and tool call instrumented with Pydantic Logfire (OpenTelemetry spans). Tracked metrics: latency_ms, tool_call_count, tools_called, tool_frequency, response_chars, BM25 top_score.

## Evaluation

58 test cases across 3 dimensions:

- **Query type** (7 values): troubleshooting, spec_lookup, setup_configuration, process_guidance, comparison_purchase, safety, weld_diagnosis
- **Welding process** (6 values): MIG, flux_cored, TIG, stick, cross_process, general
- **Difficulty** (4 values): direct_lookup, multi_hop, unanswerable, ambiguous

Data sources: 20 Reddit cases from real users (r/Welding, r/harborfreight), 30 synthetic gap-fillers, 8 adversarial hallucination traps.

```bash
python evals/run_eval.py 10                    # run 10 random cases
python evals/run_eval.py --hallucination       # run hallucination suite
python evals/run_eval.py --faithfulness        # run with LLM-as-judge scoring
```

## Project Structure

```
server.py                FastAPI server + tools + system prompt
agent.py                 CLI agent for testing
preprocess.py            PDF -> PNG + Markdown pipeline
requirements.txt         Python dependencies
railway.json             Railway deployment config
Dockerfile               Container build
knowledge_base/          Structured JSON, markdown, page images, descriptions
static/index.html        Frontend SPA
evals/
  run_eval.py            Evaluation harness
  eval_dataset.json      58 test cases with tuples and expected tools
  eval_artifacts.py      Artifact generation for eval analysis
docs/                    Design report PDF, demo video, product images
files/                   Source PDFs (owner manual, quick-start guide, selection chart)
```

## Documentation

Full design report with architecture decisions, prompt engineering details, eval methodology, and cost analysis: [Vulcan Omnipro 220 - Support Agent.pdf](docs/Vulcan%20Omnipro%20220%20-%20Support%20Agent.pdf)
