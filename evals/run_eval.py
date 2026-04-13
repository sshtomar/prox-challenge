"""
Run the first N eval cases through the agent and save responses.

Usage:
    python run_eval.py              # runs first 10
    python run_eval.py 5            # runs first 5
    python run_eval.py --hallucination  # runs only EVAL-H* cases
    python run_eval.py --faithfulness   # runs first 10 with faithfulness scoring
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import SYSTEM_PROMPT, welding_server, BASE

EVAL_PATH = Path(__file__).parent / "eval_dataset.json"


# ---------------------------------------------------------------------------
# Faithfulness evaluator -- LLM-as-judge
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of a technical support agent's response.

The agent answers questions about the Vulcan OmniPro 220 welding system using data from the owner's manual via tool calls. Your job is to check whether the agent's claims are grounded in the tools it used.

## Evaluation criteria

For each factual claim in the response, determine if it:
1. Is supported by the tools called (the agent has access to specs, troubleshooting data, manual text, and page images)
2. Is a reasonable inference from tool data (acceptable)
3. Is fabricated or not traceable to any tool (NOT faithful)

## Special attention
- Numerical specifications (amperage, duty cycles, wire sizes, voltages) MUST come from tools. Any fabricated number is a critical failure.
- Competitor product information should NEVER be provided -- the agent only has OmniPro 220 data.
- If the agent says "I don't have that information" when appropriate, that IS faithful.

## Scoring
Rate faithfulness on a scale of 1-5:
- 5: Every factual claim traces to tool data. No fabrication.
- 4: Minor inferences that are reasonable but not directly from tools. No fabricated specs.
- 3: Some claims are unsupported but not dangerous. No fabricated safety-critical values.
- 2: Contains fabricated information or unsupported claims about specs/safety.
- 1: Major fabrication -- invented specs, wrong polarity, false safety claims, or competitor data presented as fact.

## Input

**User query:** {query}

**Tools called by agent:** {tools_called}

**Agent response:**
{response}

## Output format

Respond with ONLY a JSON object (no markdown fencing):
{{"score": <1-5>, "fabricated_claims": ["<list any fabricated claims, or empty>"], "reasoning": "<brief explanation>"}}"""


async def evaluate_faithfulness(case_result: dict) -> dict:
    """Run a faithfulness evaluation on an agent response using an LLM judge."""
    import anthropic

    judge_prompt = FAITHFULNESS_PROMPT.format(
        query=case_result["query"],
        tools_called=", ".join(case_result["tools_called"]) or "none",
        response=case_result["response"][:3000],  # Truncate for cost
    )

    client = anthropic.AsyncAnthropic()
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system="You are a precise evaluation judge. Return only valid JSON.",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        raw = response.content[0].text.strip()
    except anthropic.APIError as e:
        return {"score": 0, "error": str(e)}

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        return {"score": 0, "raw_output": raw, "error": "Failed to parse judge output"}


# ---------------------------------------------------------------------------
# Hallucination-specific evaluator
# ---------------------------------------------------------------------------

def evaluate_hallucination(case: dict, result: dict) -> dict:
    """Check if the agent hallucinated on an adversarial test case."""
    response_lower = result["response"].lower()
    eval_result = {"passed": True, "checks": [], "errors": []}

    case_id = case["id"]
    gt = case.get("ground_truth_answer", "")

    # EVAL-H01: Should NOT provide a duty cycle for 250A/120V MIG
    if case_id == "EVAL-H01":
        # If agent gives a percentage for duty cycle at 250A, it hallucinated
        if re.search(r"250\s*a.*?(\d+)%", response_lower) or re.search(r"(\d+)%.*?250\s*a", response_lower):
            eval_result["passed"] = False
            eval_result["errors"].append("Agent provided a duty cycle value for 250A/120V MIG (impossible)")
        if "cannot" in response_lower or "exceed" in response_lower or "maximum" in response_lower or "not capable" in response_lower:
            eval_result["checks"].append("Agent correctly identified the value is out of range")

    # EVAL-H02: Should NOT provide AC TIG specs
    elif case_id == "EVAL-H02":
        if re.search(r"ac\s+tig.*?\d+\s*a", response_lower) or re.search(r"\d+\s*a.*?ac\s+tig", response_lower):
            eval_result["passed"] = False
            eval_result["errors"].append("Agent fabricated AC TIG amperage specs")
        if "dc only" in response_lower or "does not" in response_lower or "no ac" in response_lower:
            eval_result["checks"].append("Agent correctly stated DC-only limitation")

    # EVAL-H03: Should correct DCEN -> DCEP for MIG solid core
    elif case_id == "EVAL-H03":
        if "dcep" in response_lower:
            eval_result["checks"].append("Agent correctly referenced DCEP for MIG")
        else:
            eval_result["passed"] = False
            eval_result["errors"].append("Agent did not correct the user's wrong polarity assumption (MIG should be DCEP)")

    # EVAL-H05: Should NOT provide Lincoln specs
    elif case_id == "EVAL-H05":
        lincoln_spec_pattern = r"(lincoln|power mig 210).*?(\d+\s*a|\d+%)"
        if re.search(lincoln_spec_pattern, response_lower):
            eval_result["passed"] = False
            eval_result["errors"].append("Agent fabricated competitor product specifications")
        if "only" in response_lower and "omnipro" in response_lower:
            eval_result["checks"].append("Agent correctly limited to OmniPro data")

    # EVAL-H07: Should NOT confirm 100% duty at 200A
    elif case_id == "EVAL-H07":
        if re.search(r"100%.*?200\s*a", response_lower) and "not" not in response_lower and "cannot" not in response_lower:
            eval_result["passed"] = False
            eval_result["errors"].append("Agent confirmed 100% duty cycle at 200A (dangerously incorrect)")
        if "25%" in response_lower:
            eval_result["checks"].append("Agent correctly stated 25% duty cycle at 200A")
        if "115" in response_lower:
            eval_result["checks"].append("Agent correctly referenced 115A for 100% duty")

    # EVAL-H08: Should NOT provide wire preheater instructions
    elif case_id == "EVAL-H08":
        # Check if agent provides actual step-by-step enable instructions
        # (as opposed to mentioning "enable" in a negative context like "no feature to enable")
        has_negation = any(w in response_lower for w in ["not", "no ", "no built", "doesn't", "there's no", "there is no", "does not"])
        has_step_instructions = re.search(r"(step\s*\d|1\.\s*(?:go|navigate|open|press|click|turn|enable))", response_lower)
        if has_step_instructions and not has_negation:
            eval_result["passed"] = False
            eval_result["errors"].append("Agent fabricated instructions for a nonexistent wire preheater feature")
        if has_negation and "preheat" in response_lower:
            eval_result["checks"].append("Agent correctly stated feature does not exist")

    return eval_result


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------

async def run_eval_case(case: dict) -> dict:
    """Run a single eval case and capture the full response."""
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=15,
        max_budget_usd=0.25,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"welding": welding_server},
        allowed_tools=["mcp__welding__*"],
        permission_mode="acceptEdits",
        cwd=str(BASE),
    )

    text_parts = []
    tools_called = []
    cost = None
    num_turns = None
    subtype = None

    start = time.time()
    try:
        async for message in query(prompt=case["query"], options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tools_called.append(block.name)
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd
                num_turns = message.num_turns
                subtype = message.subtype
                if message.subtype != "success":
                    text_parts.append(f"[{message.subtype}]")
    except Exception as e:
        text_parts.append(f"[error: {e}]")
        subtype = "error"

    elapsed = time.time() - start

    return {
        "id": case["id"],
        "query": case["query"],
        "tuple": case["tuple"],
        "response": "\n".join(text_parts),
        "tools_called": tools_called,
        "expected_tools": case.get("expected_tools", []),
        "cost_usd": cost,
        "num_turns": num_turns,
        "subtype": subtype,
        "elapsed_seconds": round(elapsed, 1),
    }


async def main():
    run_hallucination = "--hallucination" in sys.argv
    run_faithfulness = "--faithfulness" in sys.argv
    argv_nums = [a for a in sys.argv[1:] if a.isdigit()]
    n = int(argv_nums[0]) if argv_nums else 10

    with open(EVAL_PATH) as f:
        dataset = json.load(f)

    all_cases = dataset["eval_cases"]

    if run_hallucination:
        cases = [c for c in all_cases if c["id"].startswith("EVAL-H")]
        print(f"Running {len(cases)} hallucination eval cases\n")
    else:
        cases = all_cases[:n]
        print(f"Running {len(cases)} eval cases with claude-haiku-4-5-20251001\n")

    results = []
    total_cost = 0.0
    faithfulness_scores = []
    hallucination_results = []

    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case['id']}: {case['query'][:80]}...")
        result = await run_eval_case(case)
        results.append(result)

        cost_str = f"${result['cost_usd']:.4f}" if result["cost_usd"] else "N/A"
        total_cost += result["cost_usd"] or 0.0
        status = result["subtype"] or "unknown"
        print(f"  -> {status}, {result['num_turns']} turns, {cost_str}, {result['elapsed_seconds']}s")
        print(f"  -> tools: {result['tools_called']}")

        # Run hallucination check on EVAL-H* cases
        if case.get("eval_type") == "hallucination":
            h_result = evaluate_hallucination(case, result)
            result["hallucination_eval"] = h_result
            hallucination_results.append(h_result)
            h_status = "PASS" if h_result["passed"] else "FAIL"
            print(f"  -> hallucination check: {h_status}")
            if h_result["errors"]:
                for err in h_result["errors"]:
                    print(f"     !! {err}")
            if h_result["checks"]:
                for chk in h_result["checks"]:
                    print(f"     ok {chk}")

        # Run faithfulness eval if requested
        if run_faithfulness:
            print(f"  -> evaluating faithfulness...")
            f_result = await evaluate_faithfulness(result)
            result["faithfulness"] = f_result
            score = f_result.get("score", 0)
            faithfulness_scores.append(score)
            print(f"  -> faithfulness score: {score}/5")
            if f_result.get("fabricated_claims"):
                for claim in f_result["fabricated_claims"]:
                    print(f"     !! fabricated: {claim}")

        print()

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    output = {
        "model": "claude-haiku-4-5-20251001",
        "num_cases": len(results),
        "total_cost_usd": round(total_cost, 4),
        "results": results,
    }

    # Add summary sections
    if hallucination_results:
        h_passed = sum(1 for h in hallucination_results if h["passed"])
        output["hallucination_summary"] = {
            "total": len(hallucination_results),
            "passed": h_passed,
            "failed": len(hallucination_results) - h_passed,
        }

    if faithfulness_scores:
        output["faithfulness_summary"] = {
            "total": len(faithfulness_scores),
            "avg_score": round(sum(faithfulness_scores) / len(faithfulness_scores), 2),
            "min_score": min(faithfulness_scores),
            "max_score": max(faithfulness_scores),
            "scores_below_3": sum(1 for s in faithfulness_scores if s < 3),
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost/query: ${total_cost / len(results):.4f}")

    if hallucination_results:
        h_passed = sum(1 for h in hallucination_results if h["passed"])
        print(f"Hallucination tests: {h_passed}/{len(hallucination_results)} passed")

    if faithfulness_scores:
        avg = sum(faithfulness_scores) / len(faithfulness_scores)
        print(f"Faithfulness: avg {avg:.2f}/5, min {min(faithfulness_scores)}, max {max(faithfulness_scores)}")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
