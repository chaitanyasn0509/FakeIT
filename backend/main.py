from fastapi import FastAPI

from backend.routes import inference
from backend.routes import training
from backend.routes import datasets
from backend.routes import satellite
from backend.routes.inference import router as inference_router
app = FastAPI(
    title="FakeIT API",
    version="1.0",
    description="AI Powered Satellite Cloud Removal Platform"
)

app.include_router(inference_router)
app.include_router(training.router)
app.include_router(datasets.router)
app.include_router(satellite.router)


@app.get("/")
def root():
    return {
        "project": "FakeIT",
        "status": "running"
    }