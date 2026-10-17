"""Test d'integration : artefacts de serving construits sur donnees factices, puis API."""
import pytest
from fastapi.testclient import TestClient

from recsys.config import load_config
from tests.conftest import COLD_CUSTOMERS


@pytest.fixture(scope="module")
def client(fake_project):
    from recsys.serving.build import build_serving

    rank = load_config("ranker")
    rank["lgbm"] |= {"n_estimators": 10, "min_child_samples": 5}
    retrievers = {}
    for name in rank["retrievers"]:
        cfg = load_config(name)
        cfg["train"] |= {"epochs": 1, "batch_size": 256}
        retrievers[name] = cfg
    build_serving(fake_project, rank, retrievers)

    from recsys.serving.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["retrievers"]


def test_recommend_known_customer(client):
    cid = client.get("/customers/sample", params={"n": 1}).json()[0]
    body = client.get(f"/recommend/{cid}", params={"k": 12}).json()
    ids = [r["article_id"] for r in body["recommendations"]]
    assert len(ids) == 12 and len(set(ids)) == 12
    assert all(r["reasons"] for r in body["recommendations"])
    assert {"retrieval_ms", "features_ms", "ranking_ms", "total_ms"} <= body["latency_ms"].keys()


def test_recommend_cold_customer(client):
    body = client.get(f"/recommend/{COLD_CUSTOMERS[0]}").json()
    assert len(body["recommendations"]) == 12


def test_unknown_customer(client):
    assert client.get("/recommend/not-a-customer").status_code == 404


def test_history_and_similar(client):
    cid = client.get("/customers/sample", params={"n": 1}).json()[0]
    hist = client.get(f"/customers/{cid}/history", params={"limit": 5}).json()
    assert hist["n_purchases"] >= 5 and len(hist["purchases"]) == 5
    article_id = hist["purchases"][0]["article_id"]
    sim = client.get(f"/articles/{article_id}/similar", params={"k": 6}).json()
    assert len(sim["similar"]) == 6
    assert article_id not in [a["article_id"] for a in sim["similar"]]
