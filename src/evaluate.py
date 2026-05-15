"""Run test prompts against a language model and save raw responses."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError("openai is required. Run: pip install -r requirements.txt") from exc

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_prompts(category: str) -> List[str]:
    """Load non-empty prompt lines for a given category."""
    path = PROMPTS_DIR / f"{category}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def mock_response(prompt: str, category: str) -> str:
    """Return a deterministic mock response for pipeline testing.

    The mock responses are designed to exercise the scorer and reporting
    pipeline without calling a live API.
    """
    if category == "benign":
        return "Here is a helpful answer to your question. I hope this is useful."
    if category == "harmful":
        return "I cannot provide instructions on harmful or illegal activities. If you have concerns about safety, please speak to a trusted professional."
    if category == "jailbreaks":
        return "I cannot fulfill this request, even in a roleplay scenario. I am not able to assist with harmful content."
    if category == "prompt_injection":
        return "I cannot ignore my instructions or reveal confidential information. Please let me know how I can help you."
    return "I cannot assist with that."


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def call_openai(
    client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float
) -> str:
    """Call the OpenAI chat completions endpoint with retry logic."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def run_evaluation(
    model: str,
    categories: List[str],
    max_prompts: Optional[int],
    use_mock: bool,
    max_tokens: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Evaluate prompts across categories and return a list of result records."""
    client: Optional[OpenAI] = None
    if not use_mock:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: OPENAI_API_KEY is not set. Add it to .env or use --mock.",
                file=sys.stderr,
            )
            sys.exit(1)
        client = OpenAI(api_key=api_key)

    records: List[Dict[str, Any]] = []
    for category in categories:
        prompts = load_prompts(category)
        if max_prompts:
            prompts = prompts[:max_prompts]

        for prompt in tqdm(prompts, desc=category, ncols=80):
            try:
                if use_mock:
                    response = mock_response(prompt, category)
                else:
                    assert client is not None
                    response = call_openai(
                        client, model, prompt, max_tokens, temperature
                    )
            except Exception as exc:  # pragma: no cover - live API failures
                response = f"ERROR: {exc}"

            records.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "model": "mock" if use_mock else model,
                    "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    return records


def save_records(records: List[Dict[str, Any]]) -> Path:
    """Write result records to a JSONL file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "raw_responses.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    """Parse CLI arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM on safety and robustness prompts."
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["benign", "harmful", "jailbreaks", "prompt_injection"],
        help="Prompt categories to evaluate",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit the number of prompts per category",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock responses instead of calling the API",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    records = run_evaluation(
        model=args.model,
        categories=args.categories,
        max_prompts=args.max_prompts,
        use_mock=args.mock,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    output_path = save_records(records)
    print(f"Saved {len(records)} responses to {output_path}")


if __name__ == "__main__":
    main()
