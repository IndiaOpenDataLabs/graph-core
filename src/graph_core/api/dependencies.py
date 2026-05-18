"""Shared FastAPI dependencies."""

import uuid

from fastapi import Header, HTTPException


async def get_namespace_id(x_namespace_id: str = Header(default="")) -> uuid.UUID:
    """Extract and validate namespace from X-Namespace-ID header."""
    if not x_namespace_id:
        raise HTTPException(status_code=400, detail="X-Namespace-ID header required")
    try:
        return uuid.UUID(x_namespace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid UUID: {x_namespace_id}")
