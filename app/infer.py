from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Tuple
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms

from helper_lib.model import SimpleCNN
from helper_lib.data_loader import cifar10_class_names

router = APIRouter(prefix="/infer", tags=["infer"])

TFM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

def load_model(weights_path: str = "./models/cifar10_cnn.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(num_classes=10)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, device

@torch.no_grad()
def predict_bytes(img_bytes: bytes, weights_path: str = "./models/cifar10_cnn.pt") -> Tuple[str, float]:
    model, device = load_model(weights_path)
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
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model weights not found. Train first.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
