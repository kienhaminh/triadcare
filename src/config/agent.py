"""Agent runtime configuration: background consultations and Care provisioning."""

from dataclasses import dataclass

from src.config._env import env_float, env_int, env_str

# Model requests of one Doctor turn, shared by the Doctor and the specialists it consults.
# When the budget holds only the closing requests, can_step() forces each running agent
# to close.
MAX_REQUESTS = 64


@dataclass(frozen=True)
class AgentSettings:
    # How long a consultation runs in the Doctor turn before it moves to the background.
    task_wait_seconds: float
    task_concurrency: int  # background consultations that run at the same time
    care_operator_token: str  # empty disables Care session provisioning
    # Model requests of one turn, shared by the Doctor and the specialists it consults.
    max_requests: int


def agent_settings() -> AgentSettings:
    return AgentSettings(
        task_wait_seconds=env_float("AGENT_TASK_WAIT_SECONDS", 20, minimum=0, maximum=600),
        task_concurrency=env_int("AGENT_TASK_CONCURRENCY", 4, minimum=1, maximum=32),
        care_operator_token=env_str("CARE_OPERATOR_TOKEN"),
        max_requests=env_int("AGENT_MAX_REQUESTS", MAX_REQUESTS, minimum=1, maximum=1024),
    )
