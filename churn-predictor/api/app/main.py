"""
main.py
=======
FastAPI application entry point.
Handles: lifespan (model + DB init), CORS, router registration, docs config.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, model, predict, customers
from app.services.db_service import init_db, seed_customers_if_empty
from app.services.model_service import ModelService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise DB tables, seed dev data, warm-up model + SHAP explainer.
    Shutdown: nothing to clean up.
    """
    log.info("=== Startup: initialising database… ===")
    await init_db()
    await seed_customers_if_empty()

    log.info("=== Startup: loading ML model… ===")
    try:
        ModelService.get()          # triggers singleton construction
        log.info("=== Model ready. ===")
    except FileNotFoundError as e:
        log.warning("Model not found: %s — /predict endpoints will return 500 until model is trained.", e)

    yield

    log.info("=== Shutdown complete. ===")


app = FastAPI(
    title="Churn Predictor API",
    description=(
        "REST API for customer churn prediction powered by XGBoost + SHAP explanations. "
        "Provides single and batch predictions, at-risk customer listing, and model metadata."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow Next.js dev server and production Firebase hosting
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.web.app",          # Firebase App Hosting
    "https://*.firebaseapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(model.router)
app.include_router(predict.router)
app.include_router(customers.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Churn Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
    }
