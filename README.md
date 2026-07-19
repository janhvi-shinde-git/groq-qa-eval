## groq-qa-eval

A minimal LLM eval harness: one test case → one API call → one pass/fail rubric score with reasoning.

### What it tests
Checks whether an LLM (Groq/Llama 3.3 70B) correctly formats a monetary "amount" field to two decimal places when instructed via prompt.

### How it works
- `PROMPT_TEMPLATE` — prompt with explicit formatting instruction
- `call_groq()` — sends prompt, gets completion
- `call_groq_mock()` — stands in for the API so scoring logic can be tested offline
- `score()` — defensive pass/fail scorer with one-line reasoning
- `run_case()` — runs a single test case end-to-end
- `main()` — entry point

### Setup
```bash
export GROQ_API_KEY="your-key-here"   # macOS/Linux
$env:GROQ_API_KEY="your-key-here"     # PowerShell, session-only
python eval.py
```

### Mock mode
No API key needed, useful for testing the scoring logic offline:
```bash
python eval.py --mock
```

### Finding
Llama 3.3 70B is **non-deterministic** on decimal formatting for this field — identical input can pass or fail across runs. Logged as an eval finding, not a bug.