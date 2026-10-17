from __future__ import annotations

from fastapi import APIRouter, Query

from recsys.serving.api.dependencies import RecommenderDep
from recsys.serving.api.schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
def health(rec: RecommenderDep) -> Health:
    return Health(
        status="ok",
        reference_date=rec.reference_date,
        retrievers=list(rec.retrievers),
        n_customers=len(rec.customer_ids),
        n_articles=rec.articles.height,
    )


@router.get("/customers/sample", response_model=list[str])
def sample_customers(
    rec: RecommenderDep, n: int = Query(8, ge=1, le=100)
) -> list[str]:
    """Clients tires au hasard parmi ceux qui ont assez d'historique (pour la demo)."""
    return rec.sample_customers(n)
