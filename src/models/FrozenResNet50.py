import torch.nn as nn
from torchvision import models

class FrozenResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for p in self.net.parameters():
            p.requires_grad = False

        # nova cabeça criada depois do congelamento.
        self.net.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.net.fc.in_features, 2),
        )


    def train(self, mode: bool = True):
        super().train(mode)

        for name, m in self.net.named_children():
            if name != "fc":
                m.eval()

        return self


    def forward(self, x):
        return self.net(x)