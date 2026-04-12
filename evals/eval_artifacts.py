"""
Artifact evaluation framework for the Prox welding agent.

Three eval layers:
  1. Structural -- Does the HTML parse and contain expected elements?
  2. Accuracy  -- Are the technical values in the artifact correct?
  3. Quality   -- LLM-as-judge for visual usefulness (optional, needs API key)

Usage:
    python evals/eval_artifacts.py                  # run all evals
    python evals/eval_artifacts.py --structural     # structural only
    python evals/eval_artifacts.py --accuracy       # accuracy only
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


# ---------------------------------------------------------------------------
# Test cases: question -> expected properties of the artifact
# ---------------------------------------------------------------------------

@dataclass
class ArtifactTestCase:
    """A test case for artifact evaluation."""
    name: str
    prompt: str
    process: str
    voltage: str
    # Structural: expect an HTML code block in the response
    expect_artifact: bool = True
    # Accuracy: strings that MUST appear somewhere in the artifact HTML
    required_content: list[str] = field(default_factory=list)
    # Accuracy: strings that must NOT appear as the *recommended* polarity.
    # Uses a regex pattern to check context -- e.g., "DCEP" near "polarity:"
    # or in a badge/label is forbidden, but "opposite of DCEP" is allowed.
    forbidden_as_recommended: list[str] = field(default_factory=list)
    # Structural: at least one of these element groups must be present.
    # Each group is a list of alternative elements (OR logic within group).
    required_element_groups: list[list[str]] = field(default_factory=list)


TEST_CASES = [
    ArtifactTestCase(
        name="TIG polarity diagram",
        prompt="Draw me a diagram showing which socket the ground clamp goes in for TIG welding.",
        process="TIG",
        voltage="240V",
        expect_artifact=True,
        required_content=[
            "Positive",       # Ground clamp goes to positive
            "Negative",       # TIG torch goes to negative
            "DCEN",           # Correct polarity name
        ],
        forbidden_as_recommended=["DCEP"],
        required_element_groups=[["svg"]],
    ),
    ArtifactTestCase(
        name="MIG polarity diagram",
        prompt="Show me a visual diagram of the cable setup for MIG solid core welding.",
        process="MIG",
        voltage="240V",
        expect_artifact=True,
        required_content=[
            "Positive",
            "Negative",
            "DCEP",
        ],
        forbidden_as_recommended=["DCEN"],
        required_element_groups=[["svg"]],
    ),
    ArtifactTestCase(
        name="Flux-Cored polarity (reversed from MIG)",
        prompt="Draw the polarity setup for flux-cored gasless welding.",
        process="Flux-Cored",
        voltage="240V",
        expect_artifact=True,
        required_content=[
            "DCEN",
            "Negative",
            "Positive",
        ],
        forbidden_as_recommended=["DCEP"],
        required_element_groups=[["svg"]],
    ),
    ArtifactTestCase(
        name="Duty cycle calculator",
        prompt="Build me an interactive duty cycle calculator for this welder.",
        process="MIG",
        voltage="240V",
        expect_artifact=True,
        required_content=[
            "200",    # 200A max for MIG 240V
            "25",     # 25% duty cycle
            "115",    # 100% duty at 115A
        ],
        # Interactive: needs at least one of input/button/select/slider
        required_element_groups=[["input", "button", "select"]],
    ),
    ArtifactTestCase(
        name="Simple spec question (should NOT generate artifact)",
        prompt="What is the maximum amperage for MIG on 240V?",
        process="MIG",
        voltage="240V",
        expect_artifact=False,
        required_content=[],
    ),
    ArtifactTestCase(
        name="Settings configurator",
        prompt="Make me an interactive tool to figure out the right settings for different material thicknesses.",
        process="MIG",
        voltage="240V",
        expect_artifact=True,
        # Needs some form of interactive UI -- any of these count
        required_element_groups=[["select", "input", "button"]],
    ),
]


# ---------------------------------------------------------------------------
# Layer 1: Structural evaluation
# ---------------------------------------------------------------------------

class HTMLValidator(HTMLParser):
    """Checks that HTML parses without errors and tracks elements found."""

    def __init__(self):
        super().__init__()
        self.elements: set[str] = set()
        self.errors: list[str] = []
        self.text_content: list[str] = []
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.elements.add(tag.lower())
        self._tag_stack.append(tag.lower())

    def handle_endtag(self, tag):
        self.elements.add(tag.lower())
        if self._tag_stack and self._tag_stack[-1] == tag.lower():
            self._tag_stack.pop()

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            self.text_content.append(stripped)

    def handle_unknown_decl(self, data):
        self.errors.append(f"Unknown declaration: {data}")


def extract_artifact_html(response_text: str) -> list[str]:
    """Extract HTML code blocks from agent response."""
    pattern = r"```html\n([\s\S]*?)```"
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    return [m.strip() for m in matches]


def eval_structural(html: str, test: ArtifactTestCase) -> dict[str, Any]:
    """Evaluate structural properties of an artifact."""
    results = {
        "parses": False,
        "element_group_checks": {},
        "errors": [],
    }

    validator = HTMLValidator()
    try:
        validator.feed(html)
        results["parses"] = True
        results["elements_found"] = sorted(validator.elements)
    except Exception as e:
        results["errors"].append(f"Parse error: {e}")
        return results

    # Check required element groups (OR logic within each group)
    for group in test.required_element_groups:
        found_any = any(elem.lower() in validator.elements for elem in group)
        group_label = " | ".join(group)
        results["element_group_checks"][group_label] = found_any
        if not found_any:
            results["errors"].append(
                f"Missing required element: need at least one of <{group_label}>"
            )

    # Check it's not trivially empty
    full_text = " ".join(validator.text_content)
    if len(full_text) < 20:
        results["errors"].append(f"Artifact has very little text content ({len(full_text)} chars)")

    return results


# ---------------------------------------------------------------------------
# Layer 2: Accuracy evaluation
# ---------------------------------------------------------------------------

def _is_recommended_polarity(html: str, polarity: str) -> bool:
    """Check if a polarity term is used as the RECOMMENDED setting, not just mentioned.

    Looks for the polarity in prominent positions:
    - In badge/label elements (large text, bold, headings)
    - As a standalone value in "Polarity: DCXX" patterns
    - In title/heading text

    Allows the polarity to appear in comparative/contrastive context like
    "opposite of DCEP" or "unlike DCEP".
    """
    p = polarity.upper()
    html_upper = html.upper()

    # Pattern 1: polarity appears in a heading, badge, or title context
    # These patterns suggest it's the recommended polarity
    recommendation_patterns = [
        rf'<H[12][^>]*>[^<]*{p}[^<]*</H[12]>',            # in headings
        rf'POLARITY[:\s]*{p}',                               # "Polarity: DCXX"
        rf'POLARITY[:\s]*</[^>]+>\s*{p}',                   # "Polarity:</span> DCXX"
        rf'{p}\s*[-—]\s*(ELECTRODE|DIRECT)',                 # "DCXX - Electrode..."
        rf'font-size:\s*(1[8-9]|[2-9]\d)px[^>]*>[^<]*{p}', # large font with polarity
        rf'font-weight:\s*(7|8|9)00[^>]*>[^<]*{p}',         # bold text with polarity
    ]

    for pattern in recommendation_patterns:
        if re.search(pattern, html_upper):
            return True

    return False


def eval_accuracy(html: str, test: ArtifactTestCase) -> dict[str, Any]:
    """Check that required values appear and forbidden polarities aren't recommended."""
    results = {
        "required_checks": {},
        "forbidden_recommendation_checks": {},
        "errors": [],
    }

    html_lower = html.lower()

    for term in test.required_content:
        found = term.lower() in html_lower
        results["required_checks"][term] = found
        if not found:
            results["errors"].append(f"Required content missing: '{term}'")

    for term in test.forbidden_as_recommended:
        is_recommended = _is_recommended_polarity(html, term)
        results["forbidden_recommendation_checks"][term] = not is_recommended
        if is_recommended:
            results["errors"].append(
                f"'{term}' appears to be RECOMMENDED as the polarity "
                f"(hallucination -- should be {test.required_content[0] if test.required_content else '?'})"
            )

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    test_name: str
    passed: bool
    artifact_generated: bool
    structural: dict[str, Any] | None = None
    accuracy: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    response_text: str = ""


async def run_agent_query(test: ArtifactTestCase) -> str:
    """Run the agent and capture the full text response."""
    # Import server tools
    from server import welding_server, SYSTEM_PROMPT

    context_prefix = f"[User has selected process: {test.process}, voltage: {test.voltage}] "
    full_prompt = context_prefix + test.prompt

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"welding": welding_server},
        allowed_tools=["mcp__welding__*"],
        permission_mode="acceptEdits",
        max_turns=10,
        cwd=str(Path(__file__).parent.parent),
    )

    response_parts = []
    async for message in query(prompt=full_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_parts.append(block.text)

    return "\n\n".join(response_parts)


async def evaluate_test(test: ArtifactTestCase, run_agent: bool = True) -> EvalResult:
    """Run a single test case through all eval layers."""
    result = EvalResult(test_name=test.name, passed=True, artifact_generated=False)

    if run_agent:
        print(f"  Running agent for: {test.name}...")
        try:
            result.response_text = await run_agent_query(test)
        except Exception as e:
            result.passed = False
            result.errors.append(f"Agent query failed: {e}")
            return result
    else:
        result.errors.append("Skipped (no agent run)")
        result.passed = False
        return result

    # Extract artifacts
    artifacts = extract_artifact_html(result.response_text)
    result.artifact_generated = len(artifacts) > 0

    # Check artifact expectation
    if test.expect_artifact and not result.artifact_generated:
        result.passed = False
        result.errors.append("Expected an artifact but none was generated")
        return result

    if not test.expect_artifact and result.artifact_generated:
        result.passed = False
        result.errors.append("Did NOT expect an artifact but one was generated (over-generating)")
        return result

    if not test.expect_artifact:
        # No artifact expected and none generated -- pass
        return result

    # Evaluate each artifact (usually just one)
    for i, html in enumerate(artifacts):
        # Structural
        structural = eval_structural(html, test)
        result.structural = structural
        if structural["errors"]:
            result.passed = False
            result.errors.extend([f"[structural] {e}" for e in structural["errors"]])

        # Accuracy
        if test.required_content or test.forbidden_as_recommended:
            accuracy = eval_accuracy(html, test)
            result.accuracy = accuracy
            if accuracy["errors"]:
                result.passed = False
                result.errors.extend([f"[accuracy] {e}" for e in accuracy["errors"]])

    return result


async def main():
    run_structural = "--structural" in sys.argv or len(sys.argv) == 1
    run_accuracy = "--accuracy" in sys.argv or len(sys.argv) == 1

    print("=" * 60)
    print("PROX ARTIFACT EVALUATION")
    print("=" * 60)
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    results = []
    for test in TEST_CASES:
        print(f"[{test.name}]")
        result = await evaluate_test(test, run_agent=True)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  Result: {status}")
        if result.artifact_generated:
            print(f"  Artifact: generated ({len(extract_artifact_html(result.response_text))} blocks)")
        else:
            print(f"  Artifact: none")
        if result.errors:
            for err in result.errors:
                print(f"  Error: {err}")
        print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")
    print("=" * 60)

    # Save detailed results
    out_path = Path(__file__).parent / "eval_results.json"
    serializable = []
    for r in results:
        serializable.append({
            "test_name": r.test_name,
            "passed": r.passed,
            "artifact_generated": r.artifact_generated,
            "structural": r.structural,
            "accuracy": r.accuracy,
            "errors": r.errors,
        })
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
