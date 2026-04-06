# server/pipeline_environment.py
import uuid
import json
from typing import Optional

import pandas as pd

from openenv.core.env_server import Environment
from openenv.core.env_server.types import Action, Observation, State

from pipeline_debugger_env.models import PipelineAction, PipelineObservation, PipelineState
from pipeline_debugger_env.server.tasks import TASKS, grade_output, safe_execute_code


class PipelineDebuggerEnvironment(Environment[PipelineAction, PipelineObservation, PipelineState]):
    """
    Data Pipeline Debugger — OpenEnv Environment

    The agent is given a broken pandas pipeline and must fix it by submitting
    corrected Python code. Graders evaluate output DataFrame correctness and
    return partial reward signals across schema, dtype, row, and value dimensions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        assert task_id in TASKS, f"task_id must be one of {list(TASKS.keys())}"
        self._task_id = task_id
        self._task = TASKS[task_id]
        self._state = PipelineState(task_id=task_id)
        self._input_data = None
        self._expected_df: Optional[pd.DataFrame] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> PipelineObservation:
        """Initialize a fresh episode."""
        self._reset_rubric()

        eid = episode_id or str(uuid.uuid4())
        self._state = PipelineState(
            episode_id=eid,
            task_id=self._task_id,
            step_count=0,
            best_score=0.0,
            cumulative_reward=0.0,
            solved=False,
        )

        self._input_data = self._task["get_input"]()
        self._expected_df = self._task["get_expected"]()

        input_str = self._serialize_input(self._input_data)
        expected_sample = self._expected_df.head(3).to_json(orient="records", date_format="iso")

        return PipelineObservation(
            task_id=self._task_id,
            task_description=self._task["description"],
            broken_code=self._task["broken_code"],
            input_data=input_str,
            expected_output_sample=expected_sample,
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
        self._state.step_count += 1
        max_steps = self._task["max_steps"]

        result_df, error_msg = safe_execute_code(
            action.code,
            self._input_data,
            self._task_id,
        )

        if error_msg:
            reward = -0.05
            self._state.cumulative_reward += reward
            done = self._state.step_count >= max_steps

            return PipelineObservation(
                task_id=self._task_id,
                task_description=self._task["description"],
                broken_code=self._task["broken_code"],
                input_data=self._serialize_input(self._input_data),
                expected_output_sample=self._expected_df.head(3).to_json(orient="records", date_format="iso"),
                last_action_result=f"EXECUTION ERROR:\n{error_msg}",
                step_count=self._state.step_count,
                done=done,
                reward=reward,
                score_breakdown=None,
            )

        score, breakdown = grade_output(result_df, self._expected_df, self._task_id)

        improvement = max(0.0, score - self._state.best_score)
        step_penalty = -0.01
        solve_bonus = 0.5 if score >= 0.95 else 0.0
        reward = round(score + improvement * 0.3 + step_penalty + solve_bonus, 4)

        self._state.best_score = max(self._state.best_score, score)
        self._state.cumulative_reward += reward
        solved = score >= 0.95
        self._state.solved = solved
        done = solved or (self._state.step_count >= max_steps)

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
            task_id=self._task_id,
            task_description=self._task["description"],
            broken_code=self._task["broken_code"],
            input_data=self._serialize_input(self._input_data),
            expected_output_sample=self._expected_df.head(3).to_json(orient="records", date_format="iso"),
            last_action_result=result_msg,
            step_count=self._state.step_count,
            done=done,
            reward=reward,
            score_breakdown=breakdown,
        )

    @property
    def state(self) -> PipelineState:
        return self._state

    def _serialize_input(self, input_data) -> str:
        if self._task["input_is_dict"]:
            return json.dumps({
                k: v.to_json(orient="records", date_format="iso")
                for k, v in input_data.items()
            })
        return input_data.to_json(orient="records", date_format="iso")

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
