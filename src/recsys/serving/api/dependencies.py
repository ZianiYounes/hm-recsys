from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from recsys.serving.recommender import Recommender


def get_recommender(request: Request) -> Recommender:
    return request.app.state.recommender


RecommenderDep = Annotated[Recommender, Depends(get_recommender)]


def resolve_customer(rec: Recommender, customer_id: str) -> int:
    c = rec.customer_idx(customer_id)
    if c is None:
        raise HTTPException(status_code=404, detail=f"Unknown customer: {customer_id}")
    return c
