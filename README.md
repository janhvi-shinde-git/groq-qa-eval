# groq-qa-eval

A minimal LLM eval harness: one test case → one API call → one pass/fail rubric score with reasoning.

## What it tests

Checks whether an LLM (Groq/Llama 3.3 70B) correctly extracts and formats structured fields — name, date, amount — from messy natural-language text, and specifically how it resolves locale-ambiguous dates (e.g. `12/06/26`) when given explicit `YYYY-MM-DD` formatting instructions.

## How it works

* `PROMPT_TEMPLATE` — prompt with explicit field and formatting instructions
* `call_groq()` — sends prompt, gets completion
* `call_groq_mock()` — stands in for the API so scoring logic can be tested offline
* `score()` — defensive pass/fail scorer with one-line reasoning
* `run_case()` — runs a single test case end-to-end
* `log_result()` — appends a run's result to `eval_log.jsonl` when `--log` is set
* `main()` — entry point; handles repeats and per-case pass-rate reporting

## Setup

```
export GROQ_API_KEY="your-key-here"   # macOS/Linux
$env:GROQ_API_KEY="your-key-here"     # PowerShell, session-only
python eval.py
```

## Mock mode

No API key needed, useful for testing the scoring logic offline:

```
python eval.py --mock
```

## Repeat mode

Reruns each case N times and reports a per-case pass rate — used to formally reproduce a claim like "10/10" instead of asserting it from memory:

```
python eval.py --repeat 10
python eval.py --mock --repeat 10
```

## Logging

Append every run's result (raw output, verdict, timestamp) to `eval_log.jsonl`:

```
python eval.py --repeat 10 --log
```

## Finding

Llama 3.3 70B has a consistent, deterministic bias toward US-style (MM/DD) interpretation of locale-ambiguous dates (e.g. `12/06/26` parsed as Dec 6 instead of June 12) — reproduced across 10/10 live API runs (`python eval.py --repeat 10 --log`) — even when the prompt explicitly requests `YYYY-MM-DD` output. The `amount` field, by contrast, was reliably correct across all runs. Logged as a model limitation, not a bug in this harness.