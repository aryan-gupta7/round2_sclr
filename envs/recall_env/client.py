from typing import Dict, Optional, List
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import RecallAction, RecallObservation, RecallState

class RecallEnv(EnvClient[RecallAction, RecallObservation, RecallState]):
    """
    Client for the Recall Environment.
    """

    def _step_payload(self, action: RecallAction) -> Dict:
        # Pydantic's model_dump is the best way to handle this
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[RecallObservation]:
        obs_data = payload.get("observation", {})
        observation = RecallObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> RecallState:
        return RecallState(**payload)
