"""HTTP helper: one polite client with retry/backoff for all providers."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

USER_AGENT = "india-rainfall-benchmark/0.1 (research; github.com/Jayesh518/india-rainfall-benchmark)"


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
def get(
    url: str,
    params: dict[str, str | float | int] | None = None,
    timeout: float = 120.0,
) -> httpx.Response:
    """GET with retries on rate limits and transient server errors."""
    resp = httpx.get(
        url,
        params=params,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    return resp


def get_once(
    url: str,
    params: dict[str, str | float | int] | None = None,
    timeout: float = 120.0,
) -> httpx.Response:
    """GET without retry or ``raise_for_status``; the caller inspects the status.

    Used where a 429 means "quota exhausted until later" rather than "try again
    in a moment", so backing off for seconds would only waste quota.
    """
    return httpx.get(
        url,
        params=params,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
