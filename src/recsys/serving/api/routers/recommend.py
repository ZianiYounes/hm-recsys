from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from recsys.serving.api.dependencies import RecommenderDep, resolve_customer
from recsys.serving.api.schemas import HistoryResponse, RecommendResponse, SimilarResponse

router = APIRouter()


@router.get("/recommend/{customer_id}", response_model=RecommendResponse, tags=["recommend"])
def recommend(
    customer_id: str,
    rec: RecommenderDep,
    k: int = Query(12, ge=1, le=50),
) -> RecommendResponse:
    c = resolve_customer(rec, customer_id)
    out = rec.recommend(c, k)
    return RecommendResponse(
        customer_id=customer_id, reference_date=rec.reference_date, **out
    )


@router.get(
    "/customers/{customer_id}/history", response_model=HistoryResponse, tags=["customers"]
)
def history(
    customer_id: str,
    rec: RecommenderDep,
    limit: int = Query(12, ge=1, le=200),
) -> HistoryResponse:
    c = resolve_customer(rec, customer_id)
    return HistoryResponse(
        customer_id=customer_id,
        n_purchases=rec.history(c).height,
        purchases=rec.purchases(c, limit),
    )


@router.get("/articles/{article_id}/similar", response_model=SimilarResponse, tags=["articles"])
def similar(
    article_id: str,
    rec: RecommenderDep,
    k: int = Query(12, ge=1, le=50),
) -> SimilarResponse:
    a = rec.article_idx(article_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"Unknown article: {article_id}")
    return SimilarResponse(article=rec.article(a), similar=rec.similar(a, k))
