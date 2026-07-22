"""
NetworkInterceptor — capture Agent A backend API responses via Playwright network hooks.

Instead of scraping Agent A's output from the DOM (which is fragile and depends on
the UI structure), this module intercepts HTTP responses at the network layer.
This is the BrowserGym-native approach: the agent has full access to the Playwright
page object, and `page.on('response')` is the standard mechanism for monitoring
backend communication.

Usage:
    from .network_interceptor import NetworkInterceptor

    interceptor = NetworkInterceptor(
        patterns=["/api/chat/**", "/api/agent/**"],
        max_body_size=50_000,
    )
    interceptor.attach(page)  # call during WorkflowTask.setup()

    # ... agent interacts with Agent A ...

    responses = interceptor.get_responses()
    interceptor.detach()  # optional cleanup
"""

import fnmatch
import json
import logging
import re
from typing import Optional

import playwright.sync_api

logger = logging.getLogger(__name__)

# Default patterns — common Agent A backend API paths.
# Users should override these with their actual Agent A API endpoints.
_DEFAULT_API_PATTERNS = [
    "/api/chat/**",
    "/api/agent/**",
    "/api/v1/chat/**",
    "/api/messages/**",
    "/v1/chat/completions",
]


class NetworkInterceptor:
    """
    Intercepts HTTP responses matching configurable URL patterns.

    Designed to capture Agent A's backend API responses (chat completions,
    analysis results, etc.) directly from network traffic, bypassing DOM scraping.

    Architecture:
        - Uses Playwright's `page.on('response', callback)` to monitor all HTTP responses
        - Filters by glob URL patterns (supports `*` and `**` wildcards)
        - Extracts and stores response body (JSON and text content types)
        - Chronological storage — responses are ordered by arrival time

    The intercepted data is compatible with the existing `agent_a_responses`
    structure consumed by EvaluatorAgent, so no changes are needed downstream.
    """

    def __init__(
        self,
        patterns: Optional[list[str]] = None,
        max_body_size: int = 50_000,
        include_headers: bool = False,
    ):
        """
        Args:
            patterns: list of glob URL patterns to match (e.g. "/api/chat/**").
                      Supports `*` (single segment) and `**` (any depth).
                      If None or empty, uses _DEFAULT_API_PATTERNS.
            max_body_size: maximum response body size in bytes (truncate beyond this).
            include_headers: if True, store response headers alongside the body.
        """
        self.patterns = patterns or _DEFAULT_API_PATTERNS
        self.max_body_size = max_body_size
        self.include_headers = include_headers
        self._responses: list[dict] = []
        self._page: Optional[playwright.sync_api.Page] = None
        self._attached = False

    def attach(self, page: playwright.sync_api.Page):
        """
        Register the response listener on a Playwright page.

        Call this during WorkflowTask.setup(), AFTER page.goto().

        Args:
            page: Playwright synchronous Page object.
        """
        if self._attached:
            logger.warning("NetworkInterceptor already attached, detaching first")
            self.detach()

        self._page = page
        self._responses = []
        page.on("response", self._on_response)
        self._attached = True
        logger.info(
            f"NetworkInterceptor attached — monitoring {len(self.patterns)} URL pattern(s): "
            f"{self.patterns}"
        )

    def detach(self):
        """Remove the response listener. Call during teardown if needed."""
        if self._page and self._attached:
            try:
                self._page.remove_listener("response", self._on_response)
            except Exception:
                pass  # Page may already be closed
            self._attached = False
            logger.info(
                f"NetworkInterceptor detached — captured {len(self._responses)} response(s)"
            )

    @property
    def is_attached(self) -> bool:
        return self._attached

    @staticmethod
    def from_env() -> "NetworkInterceptor":
        """
        Create a NetworkInterceptor from environment variables.

        Reads AGENT_A_API_PATTERNS (comma-separated glob patterns).
        Falls back to defaults if the env var is not set.

        Example:
            export AGENT_A_API_PATTERNS="/api/chat/**,/api/agent/**"
        """
        patterns_str = os.environ.get("AGENT_A_API_PATTERNS", "")
        if patterns_str:
            patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
        else:
            patterns = None  # will use defaults
        return NetworkInterceptor(patterns=patterns)

    def get_responses(self) -> list[dict]:
        """
        Return all intercepted responses in chronological order.

        Returns:
            list of dicts, each with:
                - url: str — the full response URL
                - status: int — HTTP status code
                - content_type: str — Content-Type header value
                - body_text: str — extracted text content (truncated to max_body_size)
                - body_json: dict | None — parsed JSON if applicable
                - timestamp: float — time the response was captured (seconds since epoch)
                - headers: dict | None — response headers (if include_headers=True)
                - source: "network" — always "network" (distinguishes from DOM extraction)
        """
        return list(self._responses)

    def clear(self):
        """Clear all captured responses."""
        self._responses = []

    # =========================================================================
    # Internal
    # =========================================================================

    def _on_response(self, response: playwright.sync_api.Response):
        """Callback invoked by Playwright for every HTTP response."""
        url = response.url
        if not self._matches_patterns(url):
            return

        content_type = response.headers.get("content-type", "").lower()

        # Only capture text-based responses (skip images, fonts, etc.)
        is_text = any(
            ct in content_type
            for ct in ("json", "text/", "application/javascript", "application/xml")
        )
        if not is_text:
            return

        try:
            status = response.status
            body_bytes = response.body()[: self.max_body_size]
            body_text = body_bytes.decode("utf-8", errors="replace")

            body_json = None
            if "json" in content_type:
                try:
                    body_json = json.loads(body_text)
                except (json.JSONDecodeError, ValueError):
                    pass

            entry = {
                "url": url,
                "status": status,
                "content_type": content_type,
                "body_text": body_text,
                "body_json": body_json,
                "timestamp": time.time(),
                "headers": dict(response.headers) if self.include_headers else None,
                "source": "network",
            }
            self._responses.append(entry)
            logger.debug(f"NetworkInterceptor captured: {status} {url} ({len(body_text)} bytes)")

        except Exception as e:
            logger.debug(f"NetworkInterceptor failed to read body for {url}: {e}")

    def _matches_patterns(self, url: str) -> bool:
        """Check if a URL matches any of the configured glob patterns."""
        for pattern in self.patterns:
            if fnmatch.fnmatch(url, pattern):
                return True
            # Also try matching against the URL path only (ignore scheme/host)
            try:
                from urllib.parse import urlparse
                path = urlparse(url).path
                if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
                    path, pattern.lstrip("/")
                ):
                    return True
            except Exception:
                pass
        return False


# Need these imports at module level
import os  # noqa: E402
import time  # noqa: E402