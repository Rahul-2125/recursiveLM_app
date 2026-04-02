from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
import uvicorn

app = FastAPI(
    title="RLM - Recursive Language Model",
    description="API for processing long contexts with recursive LLM queries",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "RLLM Backend Server is Active"}


if __name__ == "__main__":
    print("Starting RLM API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
