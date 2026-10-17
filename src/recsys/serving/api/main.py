"""API de recommandation.

Lancement : uvicorn recsys.serving.api.main:app --port 8000
Documentation interactive : http://localhost:8000/docs"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from recsys.config import PATHS
from recsys.serving.api.routers import health, recommend
from recsys.serving.recommender import Recommender
from recsys.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    app.state.recommender = Recommender(PATHS)
    yield


app = FastAPI(
    title="H&M Fashion Recommender",
    description="Two-stage recommender: neural retrieval (two-tower, SASRec) + LightGBM ranking.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(recommend.router)
