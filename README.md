# LLM Embedding Portal

## Project Structure
```
embedding-portal/
├── main.py                         # FastAPI application entry point
├── requirements.txt                # Project dependencies
├── .env.example                    # Example environment variables
├── controllers/
│   ├── __init__.py                 # Package initialization
│   └── embedding_controller.py     # API endpoint controller
├── services/
│   ├── __init__.py                 # Package initialization
│   ├── embedding_service.py        # Business logic service
│   └── providers/
│       ├── __init__.py             # Package initialization
│       ├── openai_provider.py      # OpenAI implementation
│       └── vertexai_provider.py    # VertexAI implementation

```