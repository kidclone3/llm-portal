from fastapi import testclient
from icecream import ic

def test_embedding(rest_client: testclient.TestClient):
    response = rest_client.post(
        "/embedding",
        json={"text": "Hello world", "embedding_model": "text-embedding-005", "provider_name": "vertexai"},
    )
    assert response.status_code == 200

    data = response.json().get("result")
    assert data.get("provider") == "vertexai"
    assert data.get("model") == "text-embedding-005"
    assert data.get("dimensions") == 768
    assert len(data.get("vector")) == 768


def test_empty_text_embedding(rest_client: testclient.TestClient):
    response = rest_client.post(
        "/embedding",
        json={"text": "", "embedding_model": "text-embedding-005", "provider_name": "vertexai"},
    )
    assert response.status_code == 400

    ic(response.json())
