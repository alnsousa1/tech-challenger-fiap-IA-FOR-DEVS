from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from womens_health_ai.api.routes import router
from womens_health_ai.core.config import api_metadata
from womens_health_ai.modeling.training import ensure_training_artifacts
from womens_health_ai.settings import paths


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_training_artifacts(force=False)
    yield


app = FastAPI(
    title=api_metadata.title,
    version=api_metadata.version,
    description=api_metadata.description,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=paths.static_dir), name="static")
app.include_router(router)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(paths.static_dir / "index.html")

