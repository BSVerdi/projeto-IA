import torch.nn as nn

class scratchCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1: 3 -> 32 canais, 224x224 -> 112x112
            self.conv_block(3, 32),

            # Bloco 2: 32 -> 64 canais, 112x112 -> 56x56
            self.conv_block(32, 64),

            # Bloco 3: 64 -> 128 canais, 56x56-> 28x28
            self.conv_block(64, 128),

            # Bloco 4: 128 -> 256 canais, 28x28 -> 14x14
            self.conv_block(128, 256),

            # Bloco 5: 256 -> 512 canais, 14x14 -> 7x7
            self.conv_block(256, 512),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )


    def conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)