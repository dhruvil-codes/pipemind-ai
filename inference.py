#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Data Pipeline Debugger (OpenEnv)

Runs a language model against all 3 tasks and reports scores.

Environment variables required:
    API_BASE_URL   — API endpoint for the LLM (OpenAI-compatible)
    MODEL_NAME     — Model identifier to use
    HF_TOKEN       — Hugging Face / API key (used as bearer token)

Usage:
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o-mini \
    HF_TOKEN=sk-... \
    python inference.py
"""

import os
import json
import time
import sys
import requests
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS = 8
MAX_TOKENS = 2048
TEMPERATURE = 0.2
TASKS = ["easy", "medium", "hard"]

# ── OpenAI client ────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
)

SYSTEM_PROMPT = """\
You are an expert Python/pandas data engineer.
You will be given a broken data pipeline and must fix it.

RULES:
1. Always define a function named `fix_pipeline` that takes the input and returns a pd.DataFrame.
2. For easy and medium tasks, signature is: def fix_pipeline(df: pd.DataFrame) -> pd.DataFrame
3. For hard task, signature is: def fix_pipeline(df: dict) -> pd.DataFrame
4. Import pandas as pd and numpy as np inside the function if needed.
5. Do NOT use print statements or side effects.
6. Return ONLY valid Python code — no markdown, no explanation, no ```python fences.

Study the broken code, understand each bug, and write a corrected version.
"""


def build_user_prompt(observation: dict, step: int, history: list[str]) -> str:
    lines = [
        f"=== TASK: {observation['task_id'].upper()} | Step {step} ===",
        "",
        "TASK DESCRIPTION:",
        observation["task_description"],
        "",
        "BROKEN CODE TO FIX:",
        observation["broken_code"],
        "",
        "INPUT DATA SAMPLE (first records):",
        observation["input_data"][:800],  # truncate for token limit
        "",
        "EXPECTED OUTPUT SAMPLE (first 3 rows):",
        observation["expected_output_sample"],
        "",
    ]
    if observation.get("last_action_result") and step > 1:
        lines += [
            "FEEDBACK FROM LAST ATTEMPT:",
            observation["last_action_result"],
            "",
        ]
    if observation.get("score_breakdown"):
        lines += [
            "SCORE BREAKDOWN:",
            json.dumps(observation["score_breakdown"], indent=2),
            "",
        ]
    if history:
        lines += ["HISTORY:", *history[-3:], ""]
    lines.append("Submit your complete fixed fix_pipeline function now:")
    return "\n".join(lines)


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_BASE_URL}{endpoint}"
    try:
        if method == "POST":
            r = requests.post(url, json=payload or {}, timeout=30)
        else:
            r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [ENV ERROR] {method} {endpoint}: {e}")
        return {}


def run_task(task_id: str) -> float:
    """Run one full episode for a task. Returns final score."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # Switch env to the desired task via query param (or env var handled server-side)
    # We hit reset with task_id in body as extra param
    reset_response = call_env("POST", "/reset", {"task_id": task_id})
    if not reset_response:
        print("  Failed to reset environment.")
        return 0.0

    observation = reset_response.get("observation", reset_response)
    history = []
    final_score = 0.0

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  Step {step}/{MAX_STEPS}")

        user_prompt = build_user_prompt(observation, step, history)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            code = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [LLM ERROR] {e}")
            code = "def fix_pipeline(df): return df"  # fallback no-op

        # Clean up any accidental markdown fences
        if "```" in code:
            lines = code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(lines)

        print(f"  Submitting code ({len(code)} chars)...")

        step_response = call_env("POST", "/step", {"code": code})
        if not step_response:
            print("  Empty step response.")
            break

        observation = step_response.get("observation", step_response)
        reward = step_response.get("reward", 0.0)
        done = step_response.get("done", False)

        result_msg = observation.get("last_action_result", "")
        score_breakdown = observation.get("score_breakdown") or {}
        last_reward = observation.get("last_reward", reward)

        # Extract score from breakdown
        if score_breakdown:
            w = {"schema_match": 0.25, "row_count_match": 0.15, "dtype_match": 0.20, "value_match": 0.40}
            final_score = sum(score_breakdown.get(k, 0) * v for k, v in w.items())

        history.append(f"Step {step}: reward={last_reward:+.3f} | {result_msg[:80]}")
        print(f"  Result: {result_msg[:100]}")
        print(f"  Reward: {last_reward:+.3f} | Done: {done}")

        if done:
            print(f"  Episode complete at step {step}.")
            break

        time.sleep(0.5)  # be gentle on the server

    print(f"\n  Final score for {task_id}: {final_score:.4f}")
    return final_score


def main():
    print("=" * 60)
    print("  Data Pipeline Debugger — Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Env:   {ENV_BASE_URL}")
    print("=" * 60)

    # Health check
    health = call_env("GET", "/health")
    if not health:
        print("\n⚠ Could not reach environment server. Is it running?")
        print(f"  Expected at: {ENV_BASE_URL}")
        sys.exit(1)

    results = {}
    for task_id in TASKS:
        score = run_task(task_id)
        results[task_id] = score
        time.sleep(1)

    print("\n" + "=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    for task_id, score in results.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id.upper():8s} [{bar}] {score:.4f}")
    avg = sum(results.values()) / len(results)
    print(f"\n  Average Score: {avg:.4f}")
    print("=" * 60)

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "scores": results, "average": avg}, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
