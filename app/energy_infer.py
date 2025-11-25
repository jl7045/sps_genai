import io
from typing import Dict
from pathlib import Path

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms

from .energy_model import get_energy_model

router = APIRouter(prefix="/energy", tags=["Energy Model"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✔ 统一从 app/models 读取
ENERGY_WEIGHTS = Path(__file__).resolve().parent / "models" / "energy_autoencoder_cifar10.pth"

ENERGY_MODEL = get_energy_model(
    device=DEVICE,
    ckpt_path=str(ENERGY_WEIGHTS),
)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


@router.post("/score")
async def energy_score(file: UploadFile = File(...)) -> Dict:
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400,
                            detail="Only JPEG/PNG images are supported.")

    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        energy = ENERGY_MODEL.energy(x)
        energy_value = float(energy.item())

    return {"energy": energy_value}
