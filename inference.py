#!/usr/bin/env python3
"""
inference.py — Baseline inference script for Data Pipeline Debugger (OpenEnv)

Runs a language model against all 3 built-in tasks and reports scores.

Environment variables required:
    API_BASE_URL   — API endpoint (default: HuggingFace Router)
    MODEL_NAME     — Model identifier to use
    HF_TOKEN       — Hugging Face token (used as bearer token for HF Router)

Usage:
    HF_TOKEN=hf_... uv run inference.py
"""

import os
import json
import time
import sys
from typing import List

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")  # Evaluators inject their own API_BASE_URL
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# Task configuration
TASK_IDS      = ["easy", "medium", "hard"]
MAX_STEPS     = 8
MAX_TOKENS    = 2048
TEMPERATURE   = 0.2
BENCHMARK     = "pipeline-debugger-env"
SUCCESS_SCORE = 0.95

# ── OpenAI client ────────────────────────────────────────────────────────────
# Client initialized inside get_model_response
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


# ── Structured Logging (Hackathon format) ────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} | env={env} | model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    action_preview = action[:100].replace("\n", "\\n")
    err_str = f" | error={error}" if error else ""
    print(f"[STEP] step={step} | action={action_preview} | reward={reward:.4f} | done={done}{err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} | steps={steps} | score={score:.4f} | rewards={rewards}", flush=True)


# ── Prompt Construction ──────────────────────────────────────────────────────
def build_user_prompt(observation: dict, step: int, history: list) -> str:
    parts = []
    parts.append(f"# Task\n{observation.get('task_description', 'Fix the pipeline')}")
    parts.append(f"\n# Broken Code\n```python\n{observation.get('broken_code', '')}\n```")

    input_data = observation.get("input_data", "")
    if len(input_data) > 1500:
        input_data = input_data[:1500] + "... (truncated)"
    parts.append(f"\n# Input Data (JSON)\n{input_data}")
    parts.append(f"\n# Expected Output Sample\n{observation.get('expected_output_sample', '')}")

    if step > 1 and observation.get("last_action_result"):
        parts.append(f"\n# Previous Result (Step {step-1})\n{observation.get('last_action_result', '')}")

    if history:
        parts.append(f"\n# History\n" + "\n".join(history[-3:]))

    parts.append(f"\n# Instructions\nReturn ONLY the Python code for fix_pipeline(). Step {step}/{MAX_STEPS}.")
    return "\n".join(parts)


def get_model_response(prompt: str) -> str:
    """Call the LLM and return generated code."""
    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", ""),
        api_key=os.environ.get("HF_TOKEN", "dummy"),
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return extract_code(text) if text else "def fix_pipeline(df): return df"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "def fix_pipeline(df): return df"


def extract_code(text: str) -> str:
    """Extract Python code from LLM response, stripping explanations and markdown."""
    import re
    # Try to extract from ```python ... ``` blocks first
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Use the longest match (most likely the complete function)
        code = max(matches, key=len).strip()
        if 'def fix_pipeline' in code:
            return code

    # If no fenced block, try to find "def fix_pipeline" directly in text
    if 'def fix_pipeline' in text:
        lines = text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith('def fix_pipeline'):
                in_function = True
            if in_function:
                # Stop if we hit a non-code line after the function body
                if code_lines and line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('def ') and not line.startswith('import ') and not line.startswith('from '):
                    break
                code_lines.append(line)
        if code_lines:
            # Also grab any imports before the function
            import_lines = [l for l in lines if l.strip().startswith(('import ', 'from ')) and lines.index(l) < lines.index(code_lines[0])]
            return '\n'.join(import_lines + code_lines).strip()

    # Fallback: return as-is (might still work)
    return text


# ── Environment API calls ────────────────────────────────────────────────────
import requests

def call_env(method: str, endpoint: str, data: dict = None) -> dict:
    """Call the environment server."""
    url = f"{ENV_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=30)
        else:
            resp = requests.post(url, json=data or {}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[DEBUG] {method} {endpoint} failed: {e}", flush=True)
        return {}


# ── Run single task ──────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    """Run one full episode for a task. Returns final score."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Reset environment with task_id
    reset_response = call_env("POST", "/reset", {"task_id": task_id})
    if not reset_response:
        print(f"[DEBUG] Failed to reset environment for task {task_id}", flush=True)
        log_end(success=False, steps=0, score=0.001, rewards=[])
        return 0.001

    observation = reset_response.get("observation", reset_response)
    history: List[str] = []
    rewards: List[float] = []
    final_score = 0.001
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        # Build prompt and get model response
        user_prompt = build_user_prompt(observation, step, history)
        code = get_model_response(user_prompt)

        # Clean up markdown fences
        if "```" in code:
            lines = code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(lines)

        # Submit to environment (OpenEnv expects {"action": {"code": ...}})
        step_response = call_env("POST", "/step", {"action": {"code": code}})
        if not step_response:
            log_step(step=step, action=code, reward=0.0, done=True, error="Empty response from /step")
            break

        observation = step_response.get("observation", step_response)
        reward = observation.get("reward", step_response.get("reward", 0.0))
        done = observation.get("done", step_response.get("done", False))

        rewards.append(reward)
        steps_taken = step

        # Extract score from breakdown
        score_breakdown = observation.get("score_breakdown") or {}
        if score_breakdown:
            w = {"schema_match": 0.25, "row_count_match": 0.15, "dtype_match": 0.20, "value_match": 0.40}
            final_score = sum(score_breakdown.get(k, 0) * v for k, v in w.items())

        result_msg = observation.get("last_action_result", "")
        history.append(f"Step {step}: reward={reward:+.3f} | {result_msg[:80]}")

        log_step(step=step, action=code, reward=reward, done=done)

        if done:
            break

        time.sleep(0.5)

    score = max(0.001, min(final_score, 0.999))
    success = score >= SUCCESS_SCORE
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────────────────
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
    all_rewards = []

    for task_id in TASK_IDS:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id.upper()}")
        print(f"{'='*60}")

        score = run_task(task_id)
        results[task_id] = score
        all_rewards.append(score)
        time.sleep(1)

    # Print summary
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for task_id, score in results.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        status = "PASS" if score >= SUCCESS_SCORE else "FAIL"
        print(f"  [{status}] {task_id:10s} [{bar}] {score:.4f}")

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"\n  Average Score: {avg:.4f}")
    print(f"{'='*60}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "benchmark": BENCHMARK,
            "scores": results,
            "average": avg,
        }, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
