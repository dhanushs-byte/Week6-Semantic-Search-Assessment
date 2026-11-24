import io
import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client

HEADERS = {"x-api-key": "test-secret"}

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_search_success(client):
    response = client.post("/search", headers=HEADERS,
        json={"query": "forgot password", "top_k": 2})
    assert response.status_code == 200
    assert "results" in response.json()

def test_search_top_k_validation(client):
    response = client.post("/search", headers=HEADERS,
        json={"query": "account help", "top_k": 21})
    assert response.status_code == 422

def test_missing_api_key(client):
    response = client.post("/search", json={"query": "account help"})
    assert response.status_code == 401

def test_invalid_api_key(client):
    response = client.post("/search", headers={"x-api-key": "wrong"},
        json={"query": "account help"})
    assert response.status_code == 403

def test_document_upload_and_index(client):
    content = b"ACCOUNT GUIDE\\nUsers can reset a forgotten password from account settings."
    response = client.post("/documents", headers=HEADERS,
        files={"file": ("guide.txt", io.BytesIO(content), "text/plain")})
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] >= 1
