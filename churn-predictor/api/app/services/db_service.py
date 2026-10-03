"""
db_service.py
=============
Async SQLAlchemy setup with SQLite for local dev.
Seeds 200 synthetic at-risk customers on first startup.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from sqlalchemy import JSON, Column, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

log = logging.getLogger(__name__)

# Default: SQLite in the repo root (overridden by DATABASE_URL env var in prod)
# On serverless platforms (e.g. Vercel), the filesystem is read-only except /tmp
_repo_db = Path(__file__).resolve().parents[3] / "churn_dev.db"
_is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
_default_path = Path("/tmp/churn_dev.db") if _is_serverless else _repo_db

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{_default_path}")

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    contract_type = Column(String, nullable=False)
    payment_method = Column(String, nullable=False)
    support_tickets_30d = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    top_factors = Column(JSON, nullable=True)


# ── DB helpers ────────────────────────────────────────────────────────────────

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables created/verified.")


async def get_session() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        yield session


async def seed_customers_if_empty() -> None:
    """Seed the DB with synthetic customer data if the table is empty."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Customer).limit(1))
        if result.first() is not None:
            log.info("DB already seeded — skipping.")
            return

    log.info("Seeding 200 synthetic customers…")
    random.seed(42)

    first_names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank",
        "Iris", "Jake", "Kim", "Liam", "Mia", "Noah", "Olivia", "Paul",
        "Quinn", "Rita", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zoe",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Wilson", "Taylor", "Anderson", "Thomas", "Jackson", "White",
        "Harris", "Martin", "Thompson", "Young", "Hall", "Robinson",
    ]
    contracts = ["month-to-month", "one_year", "two_year"]
    payments = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]

    customers = []
    for i in range(200):
        tenure = random.randint(1, 72)
        monthly = round(random.uniform(20, 120), 2)
        total = round(monthly * tenure * random.uniform(0.95, 1.05), 2)
        contract = random.choices(contracts, weights=[0.55, 0.25, 0.20])[0]
        payment = random.choice(payments)
        tickets = random.randint(0, 7)

        # Simulated churn probability (higher for risky combos)
        base_prob = 0.10
        if contract == "month-to-month":
            base_prob += 0.25
        if tenure < 12:
            base_prob += 0.20
        if tickets >= 4:
            base_prob += 0.20
        if monthly > 90:
            base_prob += 0.10
        prob = min(0.97, base_prob + random.gauss(0, 0.05))

        if prob >= 0.65:
            tier = "high"
        elif prob >= 0.35:
            tier = "medium"
        else:
            tier = "low"

        factors = [
            {"feature": "contract_type", "value": 1.0, "shap_impact": 0.42, "direction": "increases_churn"},
            {"feature": "tenure_months", "value": float(tenure), "shap_impact": -0.28, "direction": "decreases_churn"},
            {"feature": "support_tickets_30d", "value": float(tickets), "shap_impact": 0.19, "direction": "increases_churn"},
        ]

        customers.append(Customer(
            id=f"CUST-{i+1:05d}",
            name=f"{random.choice(first_names)} {random.choice(last_names)}",
            tenure_months=tenure,
            monthly_charges=monthly,
            total_charges=total,
            contract_type=contract,
            payment_method=payment,
            support_tickets_30d=tickets,
            churn_probability=round(prob, 4),
            risk_tier=tier,
            top_factors=factors,
        ))

    async with AsyncSessionLocal() as session:
        session.add_all(customers)
        await session.commit()
    log.info("Seeded %d customers.", len(customers))


async def get_at_risk_customers(limit: int = 50) -> list[Customer]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Customer)
            .order_by(Customer.churn_probability.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


async def get_customer_by_id(customer_id: str) -> Customer | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Customer).where(Customer.id == customer_id)
        )
        return result.scalar_one_or_none()
