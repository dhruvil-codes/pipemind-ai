import uuid
import json
from typing import Optional

import pandas as pd

from openenv.core.env_server import Environment
from openenv.core.env_server.types import Action, Observation, State

from models import PipelineAction, PipelineObservation, PipelineState
from server.tasks import TASKS, grade_output, safe_execute_code
from server.upload_handler import GLOBAL_TASKS_CACHE


# ─────────────────────────────────────────────────────────────────
# MODULE-LEVEL GLOBAL STATE
# OpenEnv's HTTP server creates a NEW environment instance for EVERY
# /reset and /step request, then destroys it. This means we CANNOT
# store state on `self`. We use this module-level dict instead.
# ─────────────────────────────────────────────────────────────────
_GLOBAL_STATE = {
    "task_id": None,
    "task": None,
    "input_data": None,
    "expected_df": None,
    "step_count": 0,
    "best_score": 0.0,
    "cumulative_reward": 0.0,
    "solved": False,
}


def _load_task(task_id: str):
    """Load a task from built-in TASKS or GLOBAL_TASKS_CACHE."""
    if task_id in TASKS:
        return TASKS[task_id]
    if task_id in GLOBAL_TASKS_CACHE:
        return GLOBAL_TASKS_CACHE[task_id]
    return None


class PipelineDebuggerEnvironment(Environment[PipelineAction, PipelineObservation, PipelineState]):
    """
    Data Pipeline Debugger — OpenEnv Environment

    Supports 3 built-in tasks (easy/medium/hard) AND dynamically uploaded tasks.
    Uses module-level global state to persist across OpenEnv's per-request env creation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self._task_id = task_id

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> PipelineObservation:
        """Initialize a fresh episode."""
        self._reset_rubric()

        # Accept task_id from kwargs (sent by inference.py)
        tid = kwargs.get("task_id", self._task_id)

        task = _load_task(tid)
        if not task:
            raise ValueError(
                f"Task '{tid}' not found. Available built-in: {list(TASKS.keys())}. "
                f"Dynamic cache: {list(GLOBAL_TASKS_CACHE.keys())}"
            )

        # Initialize global state
        input_data = task["get_input"]()
        expected_df = task["get_expected"]()

        _GLOBAL_STATE["task_id"] = tid
        _GLOBAL_STATE["task"] = task
        _GLOBAL_STATE["input_data"] = input_data
        _GLOBAL_STATE["expected_df"] = expected_df
        _GLOBAL_STATE["step_count"] = 0
        _GLOBAL_STATE["best_score"] = 0.0
        _GLOBAL_STATE["cumulative_reward"] = 0.0
        _GLOBAL_STATE["solved"] = False

        eid = episode_id or str(uuid.uuid4())

        return PipelineObservation(
            task_id=tid,
            task_description=task["description"],
            broken_code=task["broken_code"],
            input_data=_serialize_input(input_data, task["input_is_dict"]),
            expected_output_sample=expected_df.head(3).to_json(orient="records", date_format="iso"),
            last_action_result="Episode started. Submit your fixed fix_pipeline() code.",
            step_count=0,
            done=False,
            reward=0.0,
            score_breakdown=None,
        )

    def step(
        self,
        action: PipelineAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> PipelineObservation:
        """Execute agent's code and return graded observation."""
        # Read state from global
        task = _GLOBAL_STATE.get("task")
        if not task:
            raise ValueError("Environment not initialized. Call /reset first.")

        tid = _GLOBAL_STATE["task_id"]
        input_data = _GLOBAL_STATE["input_data"]
        expected_df = _GLOBAL_STATE["expected_df"]

        _GLOBAL_STATE["step_count"] += 1
        step_count = _GLOBAL_STATE["step_count"]
        max_steps = task["max_steps"]

        # Execute the agent's code
        result_df, error_msg = safe_execute_code(
            action.code,
            input_data,
            tid,
        )

        if error_msg:
            reward = 0.0
            _GLOBAL_STATE["cumulative_reward"] += reward
            done = step_count >= max_steps

            return PipelineObservation(
                task_id=tid,
                task_description=task["description"],
                broken_code=task["broken_code"],
                input_data=_serialize_input(input_data, task["input_is_dict"]),
                expected_output_sample=expected_df.head(3).to_json(orient="records", date_format="iso"),
                last_action_result=f"EXECUTION ERROR:\n{error_msg}",
                step_count=step_count,
                done=done,
                reward=reward,
                score_breakdown=None,
            )

        # Grade the output
        score, breakdown = grade_output(result_df, expected_df, tid)

        improvement = max(0.0, score - _GLOBAL_STATE["best_score"])
        step_penalty = -0.01
        solve_bonus = 0.5 if score >= 0.95 else 0.0
        reward = round(score + improvement * 0.3 + step_penalty + solve_bonus, 4)
        reward = max(0.0, min(reward, 1.0))  # clamp to [0, 1]

        _GLOBAL_STATE["best_score"] = max(_GLOBAL_STATE["best_score"], score)
        _GLOBAL_STATE["cumulative_reward"] += reward
        solved = score >= 0.95
        _GLOBAL_STATE["solved"] = solved
        done = solved or (step_count >= max_steps)

        result_msg = (
            f"Score: {score:.4f} | "
            f"Schema: {breakdown['schema_match']:.2f} | "
            f"Rows: {breakdown['row_count_match']:.2f} | "
            f"Dtypes: {breakdown['dtype_match']:.2f} | "
            f"Values: {breakdown['value_match']:.2f}"
        )
        if solved:
            result_msg = f"SOLVED! {result_msg}"
        elif done:
            result_msg = f"Max steps reached. {result_msg}"

        return PipelineObservation(
            task_id=tid,
            task_description=task["description"],
            broken_code=task["broken_code"],
            input_data=_serialize_input(input_data, task["input_is_dict"]),
            expected_output_sample=expected_df.head(3).to_json(orient="records", date_format="iso"),
            last_action_result=result_msg,
            step_count=step_count,
            done=done,
            reward=reward,
            score_breakdown=breakdown,
        )

    @property
    def state(self) -> PipelineState:
        return PipelineState(
            task_id=_GLOBAL_STATE.get("task_id", "easy"),
            step_count=_GLOBAL_STATE.get("step_count", 0),
            best_score=_GLOBAL_STATE.get("best_score", 0.0),
            cumulative_reward=_GLOBAL_STATE.get("cumulative_reward", 0.0),
            solved=_GLOBAL_STATE.get("solved", False),
        )

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="pipeline-debugger-env",
            description=(
                "AI agent debugs broken pandas data pipelines. "
                "3 tasks: dtype fixes (easy), null handling + aggregation (medium), "
                "multi-table join + reshape (hard)."
            ),
            version="1.0.0",
        )


def _serialize_input(input_data, is_dict: bool) -> str:
    """Serialize input data to JSON string."""
    if is_dict:
        return json.dumps({
            k: v.to_json(orient="records", date_format="iso")
            for k, v in input_data.items()
        })
    return input_data.to_json(orient="records", date_format="iso")
