import pytest
from fastapi.testclient import TestClient
from main import app  
from unittest.mock import patch

client = TestClient(app)

def test_startup():
    """Test if FAISS vector store loads successfully on startup."""
    from main import db, retriever
    assert db is not None, " FAISS database did not load properly. Check logs."
    assert retriever is not None, " Retriever did not initialize."
    print(" FAISS is loaded and retriever is ready.")


def test_query_ai_assistant_valid():
    """Test a valid query to the AI assistant."""
    payload = {"query": "What is a matrix in linear algebra?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "response" in json_data
    assert isinstance(json_data["response"], str)
    print("\nResponse:", json_data["response"])

def test_query_ai_assistant_invalid():
    """Test API behavior when PDFs are not loaded or retriever is unavailable."""
    with patch("main.db", None), patch("main.retriever", None):
        payload = {"query": "Explain eigenvectors."}
        response = client.post("/query", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "PDFs not loaded. Try restarting the server."

def test_non_academic_question():
    """Test if non-academic questions are properly redirected."""
    payload = {"query": "What is Chicken Biryani?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "response" in json_data
    assert "focus on the course" in json_data["response"]
    print("\nResponse:", json_data["response"])

@pytest.mark.parametrize("query", [
    "Explain rank of a matrix.",
    "What is the determinant?",
    "Describe Gaussian elimination.",
    "Explain the concept of eigenvalues.",
    "What is the difference between a vector space and a subspace?"
])
def test_multiple_queries(query):
    """Test multiple valid academic queries."""
    payload = {"query": query}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "response" in json_data
    assert isinstance(json_data["response"], str)
    print(f"\nQuery: {query}\nResponse: {json_data['response']}")
