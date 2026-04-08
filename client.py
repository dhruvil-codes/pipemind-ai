# client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import PipelineAction, PipelineObservation, PipelineState


class PipelineDebuggerEnv(EnvClient[PipelineAction, PipelineObservation, PipelineState]):
    """
    Client for the Data Pipeline Debugger environment.

    Async usage:
        async with PipelineDebuggerEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(PipelineAction(code="def fix_pipeline(df): ..."))

    Sync usage:
        with PipelineDebuggerEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            result = env.step(PipelineAction(code="def fix_pipeline(df): ..."))
    """

    def _step_payload(self, action: PipelineAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[PipelineObservation]:
        obs_data = payload.get("observation", payload)
        obs = PipelineObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> PipelineState:
        return PipelineState(**payload)
