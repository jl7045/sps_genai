import io
from typing import Dict

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms

from .energy_model import get_energy_model

router = APIRouter(prefix="/energy", tags=["Energy Model"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENERGY_MODEL = get_energy_model(
    device=DEVICE,
    ckpt_path="./data/energy_autoencoder_cifar10.pth",
)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # [0,1]
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
