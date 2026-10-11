"""Thin wrapper around the official TypeSafe SDK: sends one stage question
about one paper and records token usage and cost.

Two backends speak the same API:

- jev: TypeSafe's hosted Jev, paid per input token.
- local: your own decision model in llama.cpp's server, which answers the
  same requests at /v1/systemone, for free. Start it with
  `llama-server -m model.gguf --port 8080` and set DECISION_BACKEND=local.

API docs: https://docs.typesafe.ai/api
"""

import os
from dataclasses import dataclass

import requests

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

BACKENDS = ("jev", "local")
DEFAULT_LOCAL_URL = "http://127.0.0.1:8080"

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


def backend_name(backend: str | None = None) -> str:
    """The backend to use: the argument, else DECISION_BACKEND from .env, else jev."""
    name = (backend or os.getenv("DECISION_BACKEND") or "jev").strip().lower()
    if name not in BACKENDS:
        raise RuntimeError(f"Unknown decision backend {name!r}, use one of: {', '.join(BACKENDS)}")
    return name


def local_url() -> str:
    return (os.getenv("LOCAL_DECISION_URL") or DEFAULT_LOCAL_URL).strip().rstrip("/")


def check_local_server(url: str) -> None:
    """Fail early with instructions when llama.cpp's server is not running,
    instead of failing every paper of a scan one by one."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
    except requests.RequestException:
        raise RuntimeError(
            f"No local decision server at {url}. Start it with "
            f"'llama-server -m <model>.gguf --port {url.rsplit(':', 1)[-1]}', "
            "or set LOCAL_DECISION_URL in .env."
        ) from None
    if response.status_code != 200:
        raise RuntimeError(f"The local decision server at {url} is not ready yet: {response.text[:200]}")


class JevClient:
    def __init__(
        self,
        client: TypeSafeClient | None = None,
        price_per_million: float | None = None,
        backend: str | None = None,
    ):
        self.backend = backend_name(backend)
        if client is None and self.backend == "local":
            url = local_url()
            check_local_server(url)
            # llama.cpp ignores the key, but the SDK requires one.
            client = TypeSafeClient(api_key="local", base_url=url, retry=RetryPolicy(max_retries=2))
            price_per_million = 0.0
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
