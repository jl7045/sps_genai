from fastapi import FastAPI
from .infer import router as infer_router

app = FastAPI(title="SPS GenAI - Image Classifier")

@app.get("/", tags=["health"])
def root():
    return {"message": "OK"}

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

app.include_router(infer_router)

