"""Thin wrapper around the official TypeSafe SDK: sends one stage question
about one paper and records token usage and cost.

API docs: https://docs.typesafe.ai/api
"""

import os
from dataclasses import dataclass

from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafePermissionDeniedError,
    TypeSafeUnprocessableEntityError,
)

from scanner.config import Stage

# Public jev-1.13 price in USD per 1M input tokens; output tokens are free.
# https://docs.typesafe.ai/models
DEFAULT_PRICE_PER_MILLION = 0.042

# Errors that will fail for every paper (bad key, invalid stage config),
# so the scan should stop instead of skipping paper after paper.
FATAL_ERRORS = (
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
    TypeSafeBadRequestError,
    TypeSafeUnprocessableEntityError,
)


@dataclass
class JevAnswer:
    result: dict
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str


class JevClient:
    def __init__(self, client: TypeSafeClient | None = None, price_per_million: float | None = None):
        if client is None:
            if not os.getenv("TYPESAFE_API_KEY"):
                raise RuntimeError(
                    "TYPESAFE_API_KEY is not set. Copy .env.example to .env and add your key "
                    "from https://console.typesafe.ai/keys"
                )
            # The SDK retries 429/5xx with exponential backoff and honours Retry-After.
            # Big backfills can hit the rate limit for a while, so allow more retries.
            client = TypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0))
        self.client = client

        if price_per_million is None:
            price_per_million = float(
                os.getenv("JEV_PRICE_PER_MILLION_INPUT_TOKENS", DEFAULT_PRICE_PER_MILLION)
            )
        self.price_per_million = price_per_million

    def ask(self, stage: Stage, state: dict) -> JevAnswer:
        """Ask one stage's question about `state` (the paper's title and abstract)."""
        response = self.client.system_one(state, {stage.name: stage.question()})
        answer = response.answers[stage.name]

        if stage.type == "noul":
            result = {"probability": answer.noul}
        else:
            result = {
                "selected": answer.choice,
                "probabilities": dict(answer.probabilities),
                "confidence": answer.confidence,
            }

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        return JevAnswer(
            result=result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * self.price_per_million / 1_000_000,
            model=response.model,
        )

    def close(self) -> None:
        self.client.close()
