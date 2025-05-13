from fastapi import testclient
from icecream import ic

def test_embedding(rest_client: testclient.TestClient):
    response = rest_client.post(
        "/embedding",
        json={"text": "Hello world", "embedding_model": "text-embedding-005", "provider_name": "vertexai"},
    )
    assert response.status_code == 200

    data = response.json()
    ic(data)
    assert data.get("provider") == "vertexai"
    assert data.get("embedding_model") == "text-embedding-005"
    assert data.get("dimensions") == 768