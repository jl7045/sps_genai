from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Tuple
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from functools import lru_cache

from helper_lib.model import SimpleCNN
from helper_lib.data_loader import cifar10_class_names

router = APIRouter(prefix="/infer", tags=["infer"])

TFM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.pt"

@lru_cache(maxsize=1)
def load_model() -> tuple[torch.nn.Module, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(num_classes=10)
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at: {WEIGHTS_PATH}")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, device

@torch.no_grad()
def predict_bytes(img_bytes: bytes) -> Tuple[str, float]:
    model, device = load_model()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = TFM(img).unsqueeze(0).to(device)
    logits = model(x)[0]
    probs = F.softmax(logits, dim=0)
    score, idx = probs.max(dim=0)
    return cifar10_class_names()[int(idx)], float(score)

@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        label, score = predict_bytes(img_bytes)
        return {"label": label, "score": score}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
