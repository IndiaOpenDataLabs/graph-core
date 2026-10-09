"""Native llama.cpp SystemOne decision API. Never uses chat completions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx

from graph_core.provider_base_url import normalize_provider_base_url

SYSTEMONE_ENDPOINT = (
    normalize_provider_base_url("http://localhost:8081/v1/systemone")
    or "http://localhost:8081/v1/systemone"
)
SCHEMA_VERSION = "graph-decisions-v1"


class DecisionError(RuntimeError):
    """Native decision inference is unavailable or its response is invalid."""


@dataclass(frozen=True)
class Decision:
    choice: str
    probabilities: dict[str, float]
    model: str | None = None
    question_schema: dict[str, Any] | None = None

    def trace(self, task: str) -> dict[str, Any]:
        return {
            "provider": "systemone",
            "endpoint": SYSTEMONE_ENDPOINT,
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "question_schema": self.question_schema,
            "task": task,
            "choice": self.choice,
            "probabilities": self.probabilities,
        }


class SystemOneDecisionProvider:
    """Accept only native typed probability distributions; no LLM fallback.

    An optional transport is for offline tests. The endpoint is deliberately
    fixed, with the same Docker host normalization as LLM/embedding providers.
    Timeouts/errors fail the owning job rather than
    silently accepting a merge or fabricating a score.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport

    async def decide(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Decision]:
        if not questions:
            return {}
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=120, trust_env=False
            ) as client:
                response = await client.post(
                    SYSTEMONE_ENDPOINT, json={"state": state, "questions": questions}
                )
                response.raise_for_status()
                payload = response.json()
            answers = payload["answers"]
            decisions = {}
            for question_id, question in questions.items():
                options = set(question["criteria"])
                answer = answers[question_id]
                if not isinstance(answer, dict) or answer.get("type") != "choice":
                    raise ValueError("Expected a native typed choice answer")
                probabilities = answer["probabilities"]
                if not isinstance(probabilities, dict) or set(probabilities) != options:
                    raise ValueError("Option probabilities do not match the schema")
                if any(
                    isinstance(p, bool)
                    or not isinstance(p, (int, float))
                    or not math.isfinite(p)
                    or not 0 <= p <= 1
                    for p in probabilities.values()
                ):
                    raise ValueError("Probabilities must be finite values in [0, 1]")
                if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.002):
                    raise ValueError("Option probabilities must sum to one")
                choice = answer["choice"]
                if choice not in options or probabilities[choice] < max(
                    probabilities.values()
                ):
                    raise ValueError("Choice is inconsistent with its probabilities")
                decisions[question_id] = Decision(
                    choice,
                    dict(probabilities),
                    model=payload.get("model"),
                    question_schema=dict(question),
                )
            return decisions
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise DecisionError(
                f"Decision model inference failed at {SYSTEMONE_ENDPOINT}. "
                "Run a SystemOne-capable llama.cpp server with /v1/systemone; "
                "chat-generated scores are not supported."
            ) from exc
