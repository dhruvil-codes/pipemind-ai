# 🔧 Data Pipeline Debugger — OpenEnv Environment

An **OpenEnv** environment where AI agents debug broken `pandas` data transformation pipelines.
Given buggy Python code and a dataset, the agent must submit a corrected `fix_pipeline` function
that produces the expected output DataFrame.

Built for the **Meta PyTorch OpenEnv Hackathon x SST 2026**.

---

## 🎯 Why This Environment?

Every data scientist and ML engineer spends significant time debugging broken pipelines — wrong dtypes,
mishandled nulls, incorrect aggregations, broken joins. This environment formalizes that real-world task
into a structured, gradeable RL training problem.

**It is immediately useful for:**

- Evaluating LLMs on practical data engineering reasoning
- Training agents to understand pandas semantics
- Benchmarking code-generation models on correctness, not just syntax

---

## 🗂 Environment Structure

```
pipeline_debugger_env/
├── __init__.py              # Exports: PipelineAction, PipelineObservation, PipelineDebuggerEnv
├── models.py                # Typed Pydantic: Action, Observation, State
├── client.py                # PipelineDebuggerEnv (EnvClient)
├── openenv.yaml             # Environment manifest
├── pyproject.toml           # Dependencies
├── inference.py             # Baseline inference script (root level)
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI app
    ├── pipeline_environment.py  # Core Environment logic
    ├── tasks.py             # Task definitions, datasets, graders
    ├── requirements.txt
    └── Dockerfile
```

---

## ⚡ Action Space

The agent submits **Python code** as a string:

```python
PipelineAction(code="""
import pandas as pd
import numpy as np

def fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['quantity'] = df['quantity'].astype(int)
    # ... more fixes ...
    return df
""")
```

| Field  | Type  | Description                                                          |
| ------ | ----- | -------------------------------------------------------------------- |
| `code` | `str` | Python code defining `fix_pipeline(df)`. Must return `pd.DataFrame`. |

---

## 👁 Observation Space

| Field                    | Type    | Description                                                                           |
| ------------------------ | ------- | ------------------------------------------------------------------------------------- |
| `task_id`                | `str`   | `"easy"`, `"medium"`, or `"hard"`                                                     |
| `task_description`       | `str`   | Full natural language description of what to fix                                      |
| `broken_code`            | `str`   | The buggy pipeline code to fix                                                        |
| `input_data`             | `str`   | JSON-serialized input DataFrame                                                       |
| `expected_output_sample` | `str`   | First 3 rows of expected output (JSON)                                                |
| `last_action_result`     | `str`   | Execution result or error from last step                                              |
| `last_reward`            | `float` | Reward from last step                                                                 |
| `step_count`             | `int`   | Steps taken so far                                                                    |
| `done`                   | `bool`  | Episode complete flag                                                                 |
| `score_breakdown`        | `dict`  | Per-dimension scores: `schema_match`, `row_count_match`, `dtype_match`, `value_match` |

---

## 🏆 Tasks

### Task 1 — Easy: Fix Dtype Bugs

**Difficulty:** Easy | **Max steps:** 10

A sales order dataset where dtypes are all wrong:

- `quantity` cast to float instead of int
- `unit_price` cast to int (loses decimals)
- `order_date` not parsed to datetime
- `discount_pct` is `"10%"` string, not parsed to `0.10`
- `total_price` formula ignores the discount

The agent must fix all 5 bugs and compute the correct `total_price`.

**Expected baseline score:** ~0.75–0.90

---

### Task 2 — Medium: Fix Null Handling + Aggregation

**Difficulty:** Medium | **Max steps:** 15

A customer transactions dataset with:

- NaN amounts filled with mean instead of **median**
- NaN categories **dropped** instead of filled with `"unknown"`
- `-1` sentinel in `is_refund` not replaced before filling
- `order_date` not parsed; `month` hardcoded to `1`
- Aggregation doesn't filter refunds from `total_spent`

The agent must fix null handling and produce correct customer×month aggregations.

**Expected baseline score:** ~0.50–0.75

---

### Task 3 — Hard: Fix Multi-Table Join + Reshape

**Difficulty:** Hard | **Max steps:** 20

Two DataFrames (`orders` + `products`) with:

- Quantities and prices not cast from string
- INNER join used instead of **LEFT join** (drops unmatched orders)
- `month` column never extracted
- `revenue` computed with string types (crashes or wrong)
- `profit = quantity * (unit_price - cost_price)` never computed
- Groupby missing `month` dimension
- Output missing 3 required aggregation columns

The agent receives a `dict` input and must produce a correctly shaped, joined, grouped DataFrame.

**Expected baseline score:** ~0.20–0.50

---

## 🎁 Reward Function

| Signal                  | Value                                |
| ----------------------- | ------------------------------------ |
| Base score              | `0.0 – 1.0` (weighted grader output) |
| Improvement bonus       | `+0.3 × (new_score - previous_best)` |
| Step penalty            | `−0.01` per step                     |
| Solve bonus             | `+0.50` when score ≥ 0.95            |
| Execution error penalty | `−0.05`                              |

**Score breakdown weights:**

- Schema match (correct columns): **25%**
- Row count match: **15%**
- Dtype match (numeric columns): **20%**
- Value match (cell-level accuracy, 1% tolerance): **40%**

Episode ends when score ≥ 0.95 (solved) or max steps reached.

---

## 🚀 Setup & Usage

### Local (without Docker)

```bash
pip install openenv-core pandas numpy fastapi uvicorn openai

# Run server for easy task
TASK_ID=easy uvicorn pipeline_debugger_env.server.app:app --port 8000

# Run inference baseline
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=your_key \
ENV_BASE_URL=http://localhost:8000 \
python inference.py
```

### With Docker

```bash
# Build
docker build -f server/Dockerfile -t pipeline-debugger-env .

# Run (easy task)
docker run -p 8000:8000 -e TASK_ID=easy pipeline-debugger-env

# Run (hard task)
docker run -p 8000:8000 -e TASK_ID=hard pipeline-debugger-env
```

### Validate

```bash
openenv validate
```

---

## 📊 Baseline Scores

Tested with `gpt-4o-mini`:

| Task    | Score    |
| ------- | -------- |
| Easy    | 0.87     |
| Medium  | 0.63     |
| Hard    | 0.34     |
| **Avg** | **0.61** |

---

## 🔌 Client Usage

```python
import asyncio
from pipeline_debugger_env import PipelineDebuggerEnv, PipelineAction

async def main():
    async with PipelineDebuggerEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        print(obs.task_description)

        result = await env.step(PipelineAction(code="""
import pandas as pd

def fix_pipeline(df):
    df = df.copy()
    df['quantity'] = df['quantity'].astype(int)
    df['unit_price'] = df['unit_price'].astype(float)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['discount_pct'] = df['discount_pct'].str.replace('%','').astype(float) / 100
    df['total_price'] = df['quantity'] * df['unit_price'] * (1 - df['discount_pct'])
    return df
"""))
        print(f"Score breakdown: {result.observation.score_breakdown}")

asyncio.run(main())
```

---

## 📄 License

BSD 3-Clause
