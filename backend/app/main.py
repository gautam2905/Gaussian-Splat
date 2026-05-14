from fastapi import FastAPI

from .api.health import router as health_router

app = FastAPI(title="museums-backend", version="0.1.0")
app.include_router(health_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "museums-backend", "docs": "/docs"}
