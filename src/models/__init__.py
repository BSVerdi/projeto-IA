from .FrozenResNet50 import FrozenResNet50
from .ScratchCNN import ScratchCNN

MODELS = {
    "CNN": ScratchCNN,
    "ResNet50": FrozenResNet50,
}


def build_model(name: str):
    if name not in MODELS:
        raise ValueError(f"Modelo desconhecido: {name}. Opções: {list(MODELS)}")

    return MODELS[name]()