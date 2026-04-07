from __future__ import annotations

import httpx

from app.config import Settings


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    proxy = settings.https_proxy.strip() if settings.https_proxy else None
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        ),
        proxy=proxy,
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive,
        ),
    )
