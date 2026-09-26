"""
conftest.py — shared pytest fixtures for API tests.
Uses httpx.AsyncClient with ASGITransport so no server needs to be running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Make api/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.services.db_service import init_db, seed_customers_if_empty
from app.services.model_service import ModelService


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Ensure database and model are initialized before running tests."""
    await init_db()
    await seed_customers_if_empty()
    try:
        ModelService.get()
    except Exception:
        pass


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client backed by the ASGI app (no running server needed)."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Sample valid customer payload ────────────────────────────────────────────

@pytest.fixture
def valid_customer():
    return {
        "tenure_months": 24,
        "monthly_charges": 65.50,
        "total_charges": 1570.00,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "support_tickets_30d": 2,
    }
