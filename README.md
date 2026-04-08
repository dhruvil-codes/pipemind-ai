<<<<<<< HEAD
# 🧠 Pipemind AI (OpenEnv Benchmark)

Pipemind AI is a real-world, interactive AI evaluation environment built on the OpenEnv standard. It evaluates the ability of LLM-based autonomous agents to debug and correct broken pandas data transformations.

Unlike toy game environments, Pipemind AI simulates a standard workflow of a data engineer: receiving a broken pipeline script, inspecting the inputs and intended outputs, analyzing the exceptions/broken state, and producing corrected pandas code.


## 🛠️ The Environment 
The agent takes on the role of a data engineer. Each step:
1.  **Observation:** The agent receives a Pydantic `PipelineObservation` object containing the `task_id`, `broken_code`, serialized JSON `input_data`, the `expected_output_sample` schema, and performance feedback.
2.  **Action:** The agent returns a `PipelineAction` with a single `code` string field, containing a Python function `fix_pipeline()`.
3.  **Reward:** The environment securely executes the agent's code in a restricted pandas context, compares the resulting DataFrame against the hidden expected DataFrame, and returns a granular reward (0.0 to 1.0) based on schema match, data types, row counts, and strict value equality.

## 🎯 Task Progression
The environment includes three built-in deterministic tasks providing a scalable difficulty curve:

*   **Easy (`easy`): Dtype Fixer.** A broken sales pipeline pipeline where quantities exist as strings, prices are truncated, dates aren't parsed, and percentages include `%` symbols.
*   **Medium (`medium`): Null Handler & Aggregation.** Transaction data containing `NaN` gaps, dropped crucial rows, and unhandled integer sentinels (-1). Requires median imputation, logical groupings, and conditional aggregations. 
*   **Hard (`hard`): Multi-table Join & Reshape.** Simulates a relational database export. The agent receives a dictionary of two DataFrames (`orders` and `products`), and must safely `LEFT JOIN` them, calculate derived revenue/profit columns, and calculate complex group metrics by extracted datetime boundaries.

## 🚀 Running Locally

1. **Install dependencies:** 
   We recommend using `uv` or standard pip.
   ```bash
   pip install -r server/requirements.txt
   pip install -e .
   ```

2. **Start the Environment Server:**
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

3. **Run the Baseline Agent:**
   In another terminal, export your active LLM API credentials that hit the HuggingFace router (or any OpenAI-compatible proxy), and run the evaluator script:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   export API_BASE_URL="https://router.huggingface.co/v1"
   export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
   python inference.py
   ```
   
   The `inference.py` adheres strictly to the `[START]`, `[STEP]`, and `[END]` logging format demanded by the hackathon automated evaluators.

## 🐳 Docker Support
A fully working `Dockerfile` is included.
```bash
docker build -t pipemind-ai .
docker run -p 8000:8000 pipemind-ai
```

## 🧠 Action & Observation Space
*   **Type:** Text
*   **Action Space:** Takes a Python string that strictly contains `def fix_pipeline(df):` and uses standard `pd` and `np` libraries to return a mutated DataFrame.
*   **Observation Space:** Returns structured feedback `PipelineObservation`. If code execution fails, the agent receives the raw Python `traceback` in `last_action_result`. If it succeeds but is incorrect, the agent receives a fractional score breakdown explaining exactly what dimension failed (e.g. `Schema: 1.0, Values: 0.40`).

## ⚙️ Submission Details
- **Code Quality:** Type hints and Pydantic models validate seamlessly against `openenv-core`.
- **Global State Handling:** Employs a module-level dictionary to persist environment context gracefully across FastAPIs' distinct `/reset` and `/step` HTTP lifecycle.
=======
---
title: Pipemind Ai
emoji: 👁
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Interactive evaluation environment for AI Agents.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> f3ac0d3f11a63aea99fb7b191cdb41cae25df0d2
