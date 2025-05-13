import os

import fastapi
import uvicorn
from fastapi.middleware import cors
from llm_portal.entrypoints.rest import routers

def create_app():
    app = fastapi.FastAPI(
        root_path="/api/v1",
    )

    app.add_middleware(
        cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Device-Id", "Session-Id", "Authorization"],
    )
    app.include_router(routers.embedding.router)
    return app


def run():
    app = create_app()
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    run()