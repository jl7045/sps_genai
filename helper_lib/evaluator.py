import io
import torch
from PIL import Image
from torchvision import transforms
from .model import CNNModel
from .data_loader import class_names

_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)

TFM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

def load_model(weights_path="./models/cifar10_cnn.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = CNNModel()
    ckpt = torch.load(weights_path, map_location=device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval().to(device)
    return m, device

@torch.no_grad()
def predict_bytes(img_bytes: bytes, weights_path="./models/cifar10_cnn.pt"):
    model, device = load_model(weights_path)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = TFM(img).unsqueeze(0).to(device)
    logits = model(x)[0]
    probs = logits.softmax(dim=0)
    score, idx = probs.max(dim=0)
    return class_names()[int(idx)], float(score), logits.tolist()
