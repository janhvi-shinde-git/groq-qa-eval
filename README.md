## gemini-qa-eval

A minimal LLM eval harness: one test case → one API call → one pass/fail rubric score with reasoning.

### What it tests
Checks whether an LLM (currently Groq/Llama 3.3 70B) correctly formats a monetary "amount" field to two decimal places when instructed via prompt.

### How it works
- `PROMPT_TEMPLATE` — prompt with explicit formatting instruction
- `call_groq()` — sends prompt, gets completion
- `score()` — defensive pass/fail scorer with one-line reasoning
- `run_case()` — runs a single test case end-to-end
- `main()` — entry point

### Setup
```bash
$env:GROQ_API_KEY="your-key-here"   # PowerShell, session-only
python main.py
```

### Finding
Llama 3.3 70B is **non-deterministic** on decimal formatting for this field — identical input can pass or fail across runs. Logged as an eval finding, not a bug.

### Known limitation
`--mock` mode is not currently wired for Groq (only a Gemini mock exists). Not used in normal operation.