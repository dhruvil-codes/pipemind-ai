# server/app.py
import os
import io
import json
import math
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from openenv.core.env_server import create_fastapi_app

from models import PipelineAction, PipelineObservation
from server.pipeline_environment import PipelineDebuggerEnvironment
from server.upload_handler import (
    read_csv_file, read_notebook_file,
    analyze_and_generate_task, analyze_notebook_and_generate_task,
    build_dynamic_task, GLOBAL_TASKS_CACHE
)
from server.agent_runner import run_agent_on_task

TASK_ID    = os.environ.get("TASK_ID", "easy")
STATIC_DIR = Path(__file__).parent.parent / "static"

# ── JSON sanitiser — replaces NaN/Inf (not JSON-compliant) with 0 ──
def sanitize(obj):
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [sanitize(v) for v in obj]
    return obj

# In-memory session store: session_id → result
_sessions: dict = {}

# ── OpenEnv core app (provides /reset /step /state /health /ws) ──
def make_env():
    return PipelineDebuggerEnvironment(task_id=TASK_ID)

app = create_fastapi_app(make_env, PipelineAction, PipelineObservation)

# ── Static frontend ──
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

# ── Seed endpoint for evaluator (inference.py) ──
@app.post("/seed_task")
async def seed_task(
    file: UploadFile = File(...),
):
    """
    Evaluator uses this to upload a broken file and seed a dynamic OpenEnv task.
    Returns {"task_id": "uuid-here"} which can be passed to /reset.
    """
    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".csv"):
            df       = read_csv_file(content)
            task_def = analyze_and_generate_task(df, "")
            task     = build_dynamic_task(task_def, df)
        elif filename.endswith(".ipynb"):
            nb_dict, nb_code = read_notebook_file(content)
            task_def   = analyze_notebook_and_generate_task(nb_code, "")
            import numpy as _np
            df_data = {"col_a": [1,2,3], "col_b": [4,5,6], "col_c": [7,8,9]}
            df = pd.DataFrame(df_data)
            task = build_dynamic_task(task_def, df)
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .ipynb files supported")
        
        session_id = str(uuid.uuid4())
        GLOBAL_TASKS_CACHE[session_id] = task
        return {"task_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seed failed: {str(e)}")

# ── Upload endpoint ──
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    hint: str = Form(default=""),
    api_key: str = Form(default=""),
):
    """
    Accepts a CSV or .ipynb file.
    Analyzes it with GPT, generates a task, runs the agent locally, returns session result.
    """
    session_id = str(uuid.uuid4())

    if api_key:
        os.environ["HF_TOKEN"] = api_key
        import server.upload_handler as uh
        import server.agent_runner as ar
        from openai import OpenAI
        uh.HF_TOKEN = api_key
        ar.HF_TOKEN = api_key
        ar.client   = OpenAI(base_url=ar.API_BASE_URL, api_key=api_key)

    content  = await file.read()
    filename = file.filename or ""
    nb_dict = None

    try:
        if filename.endswith(".csv"):
            df       = read_csv_file(content)
            task_def = analyze_and_generate_task(df, hint)
            if not task_def.get("issues_found") and not hint.strip():
                raise HTTPException(status_code=400, detail="No missing values/data bugs found automatically.")
            task     = build_dynamic_task(task_def, df)

        elif filename.endswith(".ipynb"):
            nb_dict, nb_code = read_notebook_file(content)
            task_def   = analyze_notebook_and_generate_task(nb_code, hint)
            if not task_def.get("issues_found") and not hint.strip():
                raise HTTPException(status_code=400, detail="No logic bugs found in the notebook code automatically.")
            
            sample_cols = task_def.get("sample_columns", [])
            if not sample_cols:
                import re as _re
                code_text = task_def.get("broken_code", "") + task_def.get("correct_code", "")
                col_refs = _re.findall(r"df\[['\"](.*?)['\"]\]", code_text)
                sample_cols = list(dict.fromkeys(col_refs))
                if not sample_cols:
                    sample_cols = ["col_a", "col_b", "col_c"]

            import numpy as _np
            n_rows = 20
            df_data = {}
            for col in sample_cols:
                lc = col.lower()
                if "date" in lc or "time" in lc:
                    df_data[col] = pd.date_range("2024-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d")
                elif "id" in lc:
                    df_data[col] = list(range(1, n_rows + 1))
                elif "price" in lc or "amount" in lc or "cost" in lc:
                    df_data[col] = [str(round(v, 2)) for v in _np.random.uniform(10, 1000, n_rows)]
                else:
                    df_data[col] = [str(round(v, 2)) for v in _np.random.uniform(1, 100, n_rows)]
            df = pd.DataFrame(df_data)
            task = build_dynamic_task(task_def, df)
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .ipynb files supported")

        GLOBAL_TASKS_CACHE[session_id] = task

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Run agent synchronously
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_agent_on_task, task
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    # Build result preview
    result_df_for_preview = result.get("result_df")
    preview_rows    = []
    preview_cols    = []
    # Store notebook dict if it's a notebook upload
    if filename.endswith(".ipynb"):
        result["original_notebook"] = nb_dict
        
    if result.get("result_df") is not None:
        rdf = result["result_df"]
        result["result_csv"] = rdf.to_csv(index=False)
        display_cols = list(rdf.columns[:15])
        preview_cols = display_cols
        preview_rows = rdf[display_cols].head(10).astype(str).replace({"nan": "", "<NA>": "None", "NaN": ""}).to_dict(orient="records")
    else:
        result["result_csv"] = None
    result.pop("result_df", None)

    _sessions[session_id] = result

    return JSONResponse(sanitize({
        "session_id":      session_id,
        "success":         result["success"],
        "final_score":     result["final_score"],
        "steps_taken":     result["steps_taken"],
        "fixes_made":      result["fixes_made"],
        "issues_found":    result["issues_found"],
        "title":           result["title"],
        "difficulty":      result["difficulty"],
        "steps_log":       result["steps_log"],
        "best_code":       result.get("best_code", ""),
        "result_preview":  preview_rows,
        "result_columns":  preview_cols,
    }))


# ── Download result ──
@app.get("/download/{session_id}")
async def download_result(session_id: str, fmt: str = "csv"):
    result = _sessions.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")

    if fmt == "code" or fmt == "ipynb":
        code = result.get("best_code", "")
        if not code:
            raise HTTPException(status_code=404, detail="No fixed code available")
        
        nb = result.get("original_notebook")
        if nb:
            nb_copy = json.loads(json.dumps(nb))
            md_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Fixed Pipeline Debugger Result\n", "The code below is the fixed pipeline written by the AI agent."]
            }
            code_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.split("\n")]
            }
            nb_copy.setdefault("cells", []).extend([md_cell, code_cell])
            
            return StreamingResponse(
                io.StringIO(json.dumps(nb_copy, indent=2)),
                media_type="application/x-ipynb+json",
                headers={"Content-Disposition": f"attachment; filename=fixed_pipeline_{session_id[:8]}.ipynb"},
            )
        else:
            return StreamingResponse(
                io.StringIO(code),
                media_type="text/x-python",
                headers={"Content-Disposition": f"attachment; filename=fixed_pipeline_{session_id[:8]}.py"},
            )

    csv_data = result.get("result_csv")
    if not csv_data:
        raise HTTPException(status_code=404, detail="No result CSV available")
    return StreamingResponse(
        io.StringIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=fixed_pipeline_{session_id[:8]}.csv"},
    )


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
