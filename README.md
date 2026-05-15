# LLM Robustness and Safety Evaluation Tool

A small, self-contained toolkit for evaluating how consistently a language model refuses harmful, adversarial, and prompt-injection requests while remaining helpful on benign prompts.

This project was built as a demonstration of empirical AI safety evaluation skills: designing a test protocol, collecting model outputs, scoring them, and summarising results in a reproducible way.

## Research question

When given harmful, jailbreak-style, prompt-injection, or benign inputs, how reliably does a target LLM refuse harmful instructions and comply with legitimate ones?

## What the repo contains

- `prompts/` — four categories of test prompts:
  - `harmful.txt` — direct requests for dangerous, unethical, or illegal content
  - `jailbreaks.txt` — roleplay or framing attempts designed to bypass refusals
  - `prompt_injection.txt` — instructions that try to override system behaviour
  - `benign.txt` — ordinary, safe questions used as a control group
- `src/evaluate.py` — runs each prompt against an OpenAI model and records raw responses
- `src/scorer.py` — classifies each response as `refused`, `complied`, or `ambiguous`
- `src/report.py` — generates a Markdown report with per-category metrics and example outputs
- `notebooks/analysis.ipynb` — optional notebook for deeper inspection of results

## Quick start

1. Clone or copy this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

4. Run the evaluator:

```bash
python -m src.evaluate --model gpt-4o-mini --max-prompts 20
```

5. Score and report:

```bash
python -m src.scorer
python -m src.report
```

The final report is written to `results/report.md`.

## Mock mode

If you do not have an OpenAI API key or want to test the pipeline quickly, run with `--mock`. This uses deterministic mock responses to demonstrate the scoring and reporting flow:

```bash
python -m src.evaluate --mock
python -m src.scorer
python -m src.report
```

## Output format

After scoring, the repo produces:

- `results/raw_responses.jsonl` — model inputs and outputs, one line per prompt
- `results/classified.csv` — prompt, category, response, and classification
- `results/summary.json` — overall and per-category refusal/compliance/ambiguous rates
- `results/report.md` — human-readable summary with examples

## Scoring method

The scorer uses a simple keyword-based classifier. It looks for explicit refusal phrases (e.g., "I cannot", "I'm not able to") and compliance markers (e.g., "Here is", "Sure") to label each response. This is intentionally a baseline. A stronger evaluation would add:

- An LLM-as-judge step for borderline cases
- Sentence-level entailment checks for harmful content
- Consistency testing across paraphrased prompts
- Per-class confidence intervals

## Limitations and next steps

- The prompt sets are small and should be expanded with curated, labelled datasets.
- The scorer is keyword-based and may misclassify subtle or evasive responses.
- Only OpenAI models are supported out of the box; adding Anthropic, Google, or local models would improve coverage.
- Future work could include automated red-teaming, multishot jailbreaks, and evaluation of model explanations for refusals.

## License

MIT
