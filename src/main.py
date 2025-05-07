from fastapi import FastAPI, Depends, HTTPException
import uvicorn
from src.controllers.embedding_controller import router as embedding_router

app = FastAPI(
    title="Embedding Portal API",
    description="API for generating text embeddings from various models",
    version="1.0.0"
)

# Include routers
app.include_router(embedding_router, prefix="/api/v1", tags=["embeddings"])

@app.get("/")
def root():
    return {"message": "Welcome to Embedding Portal API. See /docs for API documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
