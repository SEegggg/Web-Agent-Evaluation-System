"""
NetworkInterceptor — capture Agent A backend API responses via Playwright network hooks.

Instead of scraping Agent A's output from the DOM (which is fragile and depends on
the UI structure), this module intercepts HTTP responses at the network layer.
This is the BrowserGym-native approach: the agent has full access to the Playwright
page object, and `page.on('response')` is the standard mechanism for monitoring
backend communication.

Supports both regular JSON responses and SSE (Server-Sent Events) streaming
responses used by Agent A's `/api/agent/chat` endpoint.

Usage:
    from .network_interceptor import NetworkInterceptor

    interceptor = NetworkInterceptor(
        patterns=["/api/agent/**", "/api/usage/**"],
        max_body_size=100_000,
    )
    interceptor.attach(page)  # call during WorkflowTask.setup()

    # ... agent interacts with Agent A ...

    responses = interceptor.get_responses()
    interceptor.detach()  # optional cleanup
"""

import fnmatch
import json
import logging
import os
import re
import time
from typing import Optional

import playwright.sync_api

logger = logging.getLogger(__name__)

# Default patterns — matches Agent A backend API (元景工业时序分析平台).
# Override via AGENT_A_API_PATTERNS env var or config.yaml `agent_a.api_patterns`.
_DEFAULT_API_PATTERNS = [
    # Core agent chat (SSE streaming)
    "/api/agent/chat",
    # Agent control
    "/api/agent/chat/stop",
    "/api/agent/chat/stream",
    "/api/agent/approve",
    "/api/agent/questions/**",
    "/api/agent/approvals/**",
    # Agent resources
    "/api/agent/sessions/**",
    "/api/agent/upload",
    "/api/agent/upload-folder",
    # Usage / context
    "/api/usage/sessions/**",
    # Auth
    "/api/auth/login",
    # Datasets (created by agent)
    "/api/datasets/**",
    # Models (created by agent)
    "/api/models/**",
    # Artifacts (created by agent)
    "/api/artifacts/**",
    # Files
    "/api/files/**",
]


class NetworkInterceptor:
    """
    Intercepts HTTP responses matching configurable URL patterns.

    Designed to capture Agent A's backend API responses (chat completions,
    analysis results, SSE streams, etc.) directly from network traffic.

    Architecture:
        - Uses Playwright's `page.on('response', callback)` to monitor all HTTP responses
        - Playwright buffers the complete response body (including SSE streams) before
          firing the callback — so we always get the full content
        - Filters by glob URL patterns (supports `*` and `**` wildcards)
        - Auto-detects SSE (text/event-stream) and parses frames
        - Extracts structured content: JSON fields, SSE assistant_text events, etc.
        - Chronological storage — responses are ordered by arrival time
    """

    # SSE content type marker
    SSE_CONTENT_TYPE = "text/event-stream"

    def __init__(
        self,
        patterns: Optional[list[str]] = None,
        max_body_size: int = 100_000,
        include_headers: bool = False,
    ):
        """
        Args:
            patterns: list of glob URL patterns to match (e.g. "/api/agent/**").
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

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def attach(self, page: playwright.sync_api.Page):
        """
        Register the response listener on a Playwright page.

        Call this during WorkflowTask.setup(), AFTER page.goto().
        Playwright buffers the complete response body (even for long SSE streams)
        before firing the callback, so `response.body()` always returns full content.

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
            f"{self.patterns[:5]}{'...' if len(self.patterns) > 5 else ''}"
        )

    def detach(self):
        """Remove the response listener. Call during teardown."""
        if self._page and self._attached:
            try:
                self._page.remove_listener("response", self._on_response)
            except Exception:
                pass
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
        Falls back to _DEFAULT_API_PATTERNS if the env var is not set.

        Example:
            export AGENT_A_API_PATTERNS="/api/agent/**,/api/usage/**"
        """
        patterns_str = os.environ.get("AGENT_A_API_PATTERNS", "")
        if patterns_str:
            patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
        else:
            patterns = None  # will use defaults
        return NetworkInterceptor(patterns=patterns)

    # =========================================================================
    # Public API
    # =========================================================================

    def get_responses(self) -> list[dict]:
        """
        Return all intercepted responses in chronological order.

        Each entry:
            - url: str — full response URL
            - status: int — HTTP status code
            - content_type: str — Content-Type header value
            - is_sse: bool — True if this is an SSE stream
            - body_text: str — raw response text (truncated to max_body_size)
            - body_json: dict | None — parsed JSON (non-SSE responses)
            - sse_events: list[dict] | None — parsed SSE frames (SSE responses only)
            - sse_text: str | None — concatenated assistant_text (SSE responses only)
            - timestamp: float — capture time (seconds since epoch)
            - headers: dict | None — response headers (if include_headers=True)
            - source: "network"
        """
        return list(self._responses)

    def clear(self):
        """Clear all captured responses."""
        self._responses = []

    # =========================================================================
    # Internal — response handler
    # =========================================================================

    def _on_response(self, response: playwright.sync_api.Response):
        """Callback invoked by Playwright for every HTTP response."""
        url = response.url
        if not self._matches_patterns(url):
            return

        content_type = response.headers.get("content-type", "").lower()

        # Accept JSON and SSE content types
        is_json = "json" in content_type
        is_sse = self.SSE_CONTENT_TYPE in content_type
        is_text = "text/" in content_type

        if not (is_json or is_sse or is_text):
            return

        try:
            status = response.status
            # Playwright buffers the complete response body before firing 'response'.
            # For SSE streams, this means body() returns the full stream text.
            body_bytes = response.body()[: self.max_body_size]
            body_text = body_bytes.decode("utf-8", errors="replace")

            entry = {
                "url": url,
                "status": status,
                "content_type": content_type,
                "is_sse": is_sse,
                "body_text": body_text,
                "body_json": None,
                "sse_events": None,
                "sse_text": None,
                "timestamp": time.time(),
                "headers": dict(response.headers) if self.include_headers else None,
                "source": "network",
            }

            if is_sse:
                entry["sse_events"], entry["sse_text"] = self._parse_sse_body(body_text)

            elif is_json:
                try:
                    entry["body_json"] = json.loads(body_text)
                except (json.JSONDecodeError, ValueError):
                    pass

            self._responses.append(entry)
            logger.debug(
                f"NetworkInterceptor captured: {status} {url} "
                f"({'SSE' if is_sse else 'JSON'}, {len(body_text)} bytes)"
            )

        except Exception as e:
            logger.debug(f"NetworkInterceptor failed to read body for {url}: {e}")

    # =========================================================================
    # SSE parsing
    # =========================================================================

    @staticmethod
    def _parse_sse_body(body: str) -> tuple[list[dict], str]:
        """
        Parse an SSE (Server-Sent Events) response body.

        Agent A SSE format (data-only, no event:/id: lines):
            data: {"type":"assistant_text","text":"我先读取文件结构。"}

            data: [DONE]

        Returns:
            (events, concatenated_text):
            - events: list of parsed SSE frames as dicts
            - concatenated_text: all assistant_text.text values joined together
        """
        events = []
        text_parts = []

        # Split by double newline (SSE frame delimiter)
        frames = body.split("\n\n")

        for frame in frames:
            frame = frame.strip()
            if not frame:
                continue

            # Extract data: lines
            data_lines = []
            for line in frame.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    data = line[5:].strip()  # Remove "data:" prefix
                    data_lines.append(data)

            if not data_lines:
                continue

            # Process data lines
            for data in data_lines:
                if data == "[DONE]":
                    events.append({"type": "done"})
                    continue

                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        evt_type = parsed.get("type", "unknown")
                        events.append(parsed)

                        # Collect assistant_text for concatenation
                        if evt_type == "assistant_text":
                            text_parts.append(parsed.get("text", ""))

                        # run_completed has a final summary
                        elif evt_type == "run_completed":
                            final = parsed.get("final", "")
                            if final:
                                text_parts.append(final)

                except (json.JSONDecodeError, ValueError):
                    # Non-JSON data line, store as-is
                    events.append({"type": "raw", "data": data})

        return events, "".join(text_parts)

    # =========================================================================
    # URL matching
    # =========================================================================

    def _matches_patterns(self, url: str) -> bool:
        """Check if a URL matches any of the configured glob patterns."""
        for pattern in self.patterns:
            # Full URL match
            if fnmatch.fnmatch(url, pattern):
                return True
            # Path-only match (ignore scheme, host, port)
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
