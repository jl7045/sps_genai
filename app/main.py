from fastapi import FastAPI
from .infer import router as infer_router

app = FastAPI(title="SPS GenAI - Image Classifier")

@app.get("/", tags=["health"])
def root():
    return {"message": "OK"}

app.include_router(infer_router)
