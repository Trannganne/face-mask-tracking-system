import torch
import torch.nn as nn

class MaskCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MaskCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192*4*4, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.5),

            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.4),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        x = self.gap(x);    x = self.fc(x)
        return x


def load_model(weight_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskCNN(num_classes=3)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    model.to(device)
    print(f"Đã load model từ: {weight_path}  |  device: {device}")
    return model, device