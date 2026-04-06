# models.py
from typing import Optional, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class PipelineAction(Action):
    """Action submitted by the agent: Python code defining fix_pipeline()."""
    code: str = Field(
        ...,
        description=(
            "Python code defining fix_pipeline(df) -> pd.DataFrame. "
            "For easy/medium tasks input is pd.DataFrame; for hard task input is dict. "
            "pandas available as pd, numpy as np."
        )
    )


class PipelineObservation(Observation):
    """
    Observation returned after each step.
    Inherits `done`, `reward`, `metadata` from base Observation.
    """
    task_id: str = Field(..., description="Current task: easy | medium | hard")
    task_description: str = Field(..., description="What the agent must fix")
    broken_code: str = Field(..., description="The broken pipeline code to fix")
    input_data: str = Field(..., description="JSON-serialized input DataFrame (records orientation)")
    expected_output_sample: str = Field(..., description="First 3 rows of expected output (JSON)")
    last_action_result: Optional[str] = Field(None, description="Execution result or error from last step")
    step_count: int = Field(0, description="Steps taken so far this episode")
    score_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Per-dimension scores: schema_match, row_count_match, dtype_match, value_match"
    )


class PipelineState(State):
    """Episode-level state. Inherits episode_id, step_count from base State."""
    task_id: str = Field("easy", description="Current task being evaluated")
    best_score: float = Field(0.0, description="Best score achieved this episode")
    cumulative_reward: float = Field(0.0, description="Total reward accumulated this episode")
    solved: bool = Field(False, description="Whether score >= 0.95 was reached")
