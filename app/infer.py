from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from helper_lib.evaluator import predict_bytes

router = APIRouter(prefix="/v1", tags=["classifier"])

class PredictResponse(BaseModel):
    top1_class: str
    top1_score: float
    logits: List[float]

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        cls, score, logits = predict_bytes(data, "./models/cifar10_cnn.pt")
        return PredictResponse(top1_class=cls, top1_score=score, logits=logits)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to classify image: {e}")
