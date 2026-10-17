from __future__ import annotations

from pydantic import BaseModel


class Article(BaseModel):
    article_id: str
    name: str | None
    product_type: str | None
    colour: str | None
    department: str | None
    index_group: str | None
    image_path: str


class Recommendation(Article):
    rank: int
    score: float
    reasons: list[str]


class RecommendResponse(BaseModel):
    customer_id: str
    reference_date: str
    n_candidates: int
    latency_ms: dict[str, float]
    recommendations: list[Recommendation]


class Purchase(Article):
    date: str
    price: float


class HistoryResponse(BaseModel):
    customer_id: str
    n_purchases: int
    purchases: list[Purchase]


class SimilarResponse(BaseModel):
    article: Article
    similar: list[Article]


class Health(BaseModel):
    status: str
    reference_date: str
    retrievers: list[str]
    n_customers: int
    n_articles: int
