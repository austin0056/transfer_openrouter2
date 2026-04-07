from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


async def verify_gateway_key(
    authorization: str | None = Header(None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.gateway_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GATEWAY_API_KEY is not configured",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization[7:].strip()
    if token != settings.gateway_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
