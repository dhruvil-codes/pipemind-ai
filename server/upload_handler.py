# server/upload_handler.py
import json, re, io, os
import pandas as pd
import numpy as np
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

def get_client():
    return OpenAI(base_url=API_BASE_URL, api_key=os.environ.get("HF_TOKEN", HF_TOKEN) or "dummy")

def read_csv_file(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

def read_notebook_file(content: bytes):
    try:
        nb = json.loads(content.decode("utf-8"))
    except:
        nb = {"cells": []}
    cells = nb.get("cells", [])
    code_cells = [
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    ]
    return nb, "\n\n# --- next cell ---\n\n".join(code_cells)

def analyze_and_generate_task(df: pd.DataFrame, hint: str = "") -> dict:
    # ── Truncate to stay well under 128k token limit ──────────────
    MAX_COLS = 30
    all_cols  = list(df.columns)
    cols      = all_cols[:MAX_COLS]          # cap columns
    df_trim   = df[cols]                     # work on trimmed df
    dtypes    = {col: str(dt) for col, dt in df_trim.dtypes.items()}
    nulls     = {col: int(v) for col, v in df_trim.isnull().sum().items()}
    sample    = df_trim.head(3).to_json(orient="records", date_format="iso")[:3000]  # cap sample
    shape     = {"rows": len(df), "cols": len(all_cols)}
    omitted   = f" (showing first {MAX_COLS} of {len(all_cols)})" if len(all_cols) > MAX_COLS else ""

    prompt = f"""You are a senior data engineer. Analyze this raw dataset and create a data cleaning task.

DATASET:
- Shape: {shape}
- Columns{omitted}: {cols}
- Dtypes: {json.dumps(dtypes)[:1000]}
- Null counts: {json.dumps(nulls)[:500]}
- Sample (3 rows): {sample}
- User hint: "{hint[:200]}"

TASK: Identify REAL data quality issues (missing values, bad formatting, outliers, weird types) in this dataset. Then write the clean, production-ready \`correct_code\` pipeline that fixes these exact issues. 

Return ONLY valid JSON (no markdown, no backticks):
{{
  "task_id": "clean_csv",
  "title": "Clean Data Issues",
  "description": "2-3 sentences describing the data anomalies found and what needs to be cleaned.",
  "issues_found": ["Missing values in col X", "String formatting inconsistency in col Y", "Dates not parsed in col Z"],
  "broken_code": "import pandas as pd\\nimport numpy as np\\n\\ndef fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:\\n    df = df.copy()\\n    # TODO: Implement cleaning logic\\n    return df",
  "correct_code": "import pandas as pd\\nimport numpy as np\\n\\ndef fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:\\n    df = df.copy()\\n    # exact pandas logic fixing the above issues\\n    return df",
  "difficulty": "medium"
}}

CRITICAL RULES:
- ONLY use column names from this list: {cols}
- `broken_code` MUST be a blank template `def fix_pipeline(df): return df.copy()`
- `correct_code` must fix all identified anomalies and return a clean DataFrame
- Keep it practical and robust
- DO NOT invent arbitrary sentinel values like `-1` or `'Not Started'`. If a column has NaNs because it resembles an Excel grouped layout (e.g. `Week 1`, `NaN`, `NaN`), use forward-fill (`.ffill()`).
- DO NOT drastically delete columns or rows. Your job is to format the data properly and correct mistakes, not change the entire structure.
- If there are NO obvious issues and the user hint is empty, return an empty `issues_found` array `[]` and DO NOT invent fake issues."""

    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content or ""
    # Strip any markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # Extract JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)

def analyze_notebook_and_generate_task(notebook_code: str, hint: str = "") -> dict:
    # Cap notebook code to ~2000 chars to avoid context overflow
    code_sample = notebook_code[:2000]

    prompt = f"""You are a senior data engineer. Analyze this Jupyter notebook and create a pipeline debugging task.

NOTEBOOK CODE:
{code_sample}

User hint: "{hint[:200]}"

Return ONLY valid JSON (no markdown, no backticks):
{{
  "task_id": "user_upload",
  "title": "short title based on what notebook does",
  "description": "2-3 sentences about this pipeline and its bugs",
  "issues_found": ["bug 1", "bug 2", "bug 3"],
  "sample_columns": ["exact_col_1", "exact_col_2", "exact_col_3"],
  "broken_code": "import pandas as pd\\nimport numpy as np\\n\\ndef fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:\\n    df = df.copy()\\n    # broken code here\\n    return df",
  "correct_code": "import pandas as pd\\nimport numpy as np\\n\\ndef fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:\\n    df = df.copy()\\n    # correct code here\\n    return df",
  "difficulty": "medium"
}}

CRITICAL RULES:
- sample_columns MUST list EVERY column name that broken_code and correct_code reference via df['col']
- broken_code and correct_code MUST ONLY use columns listed in sample_columns
- Extract REAL pipeline logic from the notebook code
- broken_code must run without crashing but give wrong results
- correct_code fixes all bugs
- Both must define fix_pipeline(df) -> pd.DataFrame
- If there are NO obvious issues and the user hint is empty, return an empty `issues_found` array `[]`."""

    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content or ""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)

def build_dynamic_task(task_def: dict, input_df: pd.DataFrame) -> dict:
    correct_code = task_def["correct_code"]
    local_ns     = {"pd": pd, "np": np}
    correct_code_works = False

    try:
        exec(compile(correct_code, "<correct>", "exec"), local_ns)
        fix_fn = local_ns.get("fix_pipeline")
        if not fix_fn:
            raise ValueError("fix_pipeline not defined in correct_code")
        expected_df = fix_fn(input_df.copy())
        if not isinstance(expected_df, pd.DataFrame):
            raise ValueError("correct_code must return DataFrame")
        correct_code_works = True
    except Exception as e:
        # Fallback: expected = input (at least schema matches)
        print(f"[build_dynamic_task] correct_code failed: {e}")
        expected_df = input_df.copy()

    diff_map   = {"easy": 10, "medium": 15, "hard": 20}
    difficulty = task_def.get("difficulty", "medium")
    _inp       = input_df.copy()
    _exp       = expected_df.copy()

    desc = task_def.get("description", "Fix the data pipeline.")
    if not correct_code_works:
        desc += f"\n\nNote: The reference solution had errors. Available columns are: {list(input_df.columns)}"

    return {
        "task_id":       "user_upload",
        "description":   desc,
        "title":         task_def.get("title", "User Upload Task"),
        "broken_code":   task_def["broken_code"],
        "correct_code":  correct_code,
        "issues_found":  task_def.get("issues_found", []),
        "difficulty":    difficulty,
        "get_input":     lambda: _inp.copy(),
        "get_expected":  lambda: _exp.copy(),
        "max_steps":     diff_map.get(difficulty, 15),
        "input_is_dict": False,
        "expected_df":   _exp,
    }

