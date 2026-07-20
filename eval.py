"""
groq-qa-eval — smallest working loop.

Task: structured data extraction from messy text (name, date, amount).
Flow: one input -> Groq (Llama 3.3 70B) call -> parse JSON -> score vs. expected -> pass/fail + reason.

Usage:
    export GROQ_API_KEY="your-key-here"
    python eval.py

Mock mode (no API key needed, for testing the scoring logic offline):
    python eval.py --mock

Format all monetary amounts as strings with exactly two decimal places (e.g. "180.00", not "180" or 180.0).
"""

import os
import sys
import json
import re

MOCK_MODE = "--mock" in sys.argv

# ---- 1. Test cases ----
# Each case: a messy input + the ground truth we already know is correct.
# Built from the 4 exploratory prompts run manually in AI Studio.
PROMPT_TEMPLATE = """Extract the following fields from the text below and return ONLY a JSON object,
no markdown fences, no explanation.

Fields:
- name: full name of the customer
- date: in YYYY-MM-DD format
- amount: numeric value only, no currency symbol, no words

Rules:
- If a field is genuinely not present in the text, use the string "not found" — never invent a value.
- If multiple transactions are mentioned (e.g. a payment AND a refund/adjustment), extract only
  the PRIMARY payment. Ignore refunds, adjustments, or corrections.
- Format amount as a string with exactly two decimal places (e.g. "180.00", not "180" or 180.0).

Text:
\"\"\"{input_text}\"\"\"

Return JSON like: {{"name": "...", "date": "...", "amount": "..."}}
"""

TEST_CASES = [
    {
        "id": "clean",
        "input_text": (
            "Hi Priya Sharma, we've received your payment of $180.00 "
            "on June 12, 2026. Thank you!"
        ),
        "expected": {"name": "Priya Sharma", "date": "2026-06-12", "amount": "180.00"},
    },
    {
        # Renamed from "ambiguous_date_amount" — the amount field is never ambiguous
        # here (it passes 10/10 live runs); the actual finding is a deterministic
        # MM/DD-first bias on locale-ambiguous dates, even with explicit
        # YYYY-MM-DD formatting instructions in the prompt.
        "id": "ambiguous_date_format",
        "input_text": (
            "Payment from R. Mehta processed 12/06/26 for one hundred and eighty dollars."
        ),
        "expected": {"name": "R. Mehta", "date": "2026-06-12", "amount": "180.00"},
    },
    {
        "id": "missing_fields",
        "input_text": (
            "Thanks for your order! Your payment was processed on June 12, 2026. "
            "We appreciate your business."
        ),
        "expected": {"name": "not found", "date": "2026-06-12", "amount": "not found"},
    },
    {
        "id": "conflicting_payment_refund",
        "input_text": (
            "Hi Priya Sharma, your payment of $180.00 was processed on June 12, 2026. "
            "Note: refund of $180.00 issued on June 15, 2026 to Priya S. Sharma due to "
            "duplicate charge."
        ),
        "expected": {"name": "Priya Sharma", "date": "2026-06-12", "amount": "180.00"},
    },
]


def call_groq(prompt: str) -> str:
    """Real API call to Groq (Llama 3.3 70B). Requires GROQ_API_KEY env var."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Run: $env:GROQ_API_KEY='your-key-here'"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


MOCK_RESPONSES = {
    "clean": '{"name": "Priya Sharma", "date": "2026-06-12", "amount": "180.00"}',
    "ambiguous_date_format": '{"name": "R. Mehta", "date": "2026-06-12", "amount": "180.00"}',
    "missing_fields": '{"name": "not found", "date": "2026-06-12", "amount": "not found"}',
    "conflicting_payment_refund": '{"name": "Priya Sharma", "date": "2026-06-12", "amount": "180.00"}',
}


def call_groq_mock(prompt: str, case_id: str = "clean") -> str:
    """Stands in for the Groq API so we can test parsing/scoring without a key or network."""
    return MOCK_RESPONSES[case_id]


def parse_response(raw_text: str) -> dict:
    """Llama sometimes wraps JSON in ```json fences even when told not to. Strip them."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(cleaned)


def score(actual: dict, expected: dict) -> dict:
    """Field-by-field pass/fail, plus overall verdict and a one-line reason."""
    field_results = {}
    for field, expected_val in expected.items():
        actual_val = str(actual.get(field, "")).strip()
        field_results[field] = (actual_val == expected_val)

    overall_pass = all(field_results.values())

    if overall_pass:
        reason = "All fields matched expected values exactly."
    else:
        failed_fields = [f for f, ok in field_results.items() if not ok]
        details = ", ".join(
            f"{f} (got '{actual.get(f, '')}', expected '{expected[f]}')"
            for f in failed_fields
        )
        reason = f"Mismatch on: {details}"

    return {
        "field_results": field_results,
        "overall_pass": overall_pass,
        "reason": reason,
    }


def run_case(case: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(input_text=case["input_text"])

    if MOCK_MODE:
        raw = call_groq_mock(prompt, case_id=case["id"])
    else:
        raw = call_groq(prompt)

    try:
        actual = parse_response(raw)
    except json.JSONDecodeError as e:
        return {
            "id": case["id"],
            "overall_pass": False,
            "reason": f"Could not parse model output as JSON ({e})",
            "raw": raw,
        }

    result = score(actual, case["expected"])
    result["id"] = case["id"]
    result["raw"] = raw
    result["actual"] = actual
    result["expected"] = case["expected"]
    return result


def main():
    print("=" * 60)
    print("groq-qa-eval — run")
    print("=" * 60)
    if MOCK_MODE:
        print("[MOCK MODE — no real API calls made]")

    results = []
    for case in TEST_CASES:
        print(f"\n--- Case: {case['id']} ---")
        print(f"Input:    {case['input_text']}")
        result = run_case(case)
        print(f"Raw:      {result['raw']}")
        if "actual" in result:
            print(f"Actual:   {result['actual']}")
            print(f"Expected: {result['expected']}")
        verdict = "PASS" if result["overall_pass"] else "FAIL"
        print(f"[{verdict}] {result['reason']}")
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["overall_pass"])
    for r in results:
        verdict = "PASS" if r["overall_pass"] else "FAIL"
        print(f"  [{verdict}] {r['id']}")
    print(f"\n{passed}/{len(results)} cases passed.")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()