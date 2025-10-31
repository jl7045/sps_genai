import base64
from io import BytesIO
import torch
from fastapi import APIRouter
from PIL import Image
from .gan_models import Generator

router = APIRouter(prefix="/gan", tags=["gan"])

MODEL_PATH = "models/gan_generator.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_generator(weights_path=MODEL_PATH):
    ckpt = torch.load(weights_path, map_location=DEVICE)
    z_dim = ckpt["z_dim"]
    G = Generator(noise_dim=z_dim).to(DEVICE)
    G.load_state_dict(ckpt["state_dict"])
    G.eval()
    return G, z_dim

G_MODEL, Z_DIM = load_generator()

def tensor_to_pil(img_tensor):
    img = (img_tensor.squeeze(0).detach().cpu() + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = (img * 255).to(torch.uint8).numpy()
    pil_img = Image.fromarray(img, mode="L")
    return pil_img

@router.get("/generate")
def generate_digit():
    with torch.no_grad():
        noise = torch.randn(1, Z_DIM, device=DEVICE)
        fake_img = G_MODEL(noise)[0]
    pil_img = tensor_to_pil(fake_img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {"image_base64_png": img_b64}
