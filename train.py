import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# ============================
# Paths & Device
# ============================
data_dir = "dataset/rice_train"
model_path = "model/rice_effnet.pth"
os.makedirs("model", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📦 Using device:", device)

# ============================
# Data Augmentation (improved)
# ============================
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # Required for EfficientNet
])

print("📁 Loading dataset...")
full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
num_classes = len(full_dataset.classes)

# ===== Split into Train / Validation (90/10) =====
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

print(f"✅ Total: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")
print("Classes:", full_dataset.classes)

# ============================
# Mixup Function
# ============================
def mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, 1
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ============================
# Load EfficientNet B0
# ============================
print("🧠 Loading EfficientNet-B0...")
model = models.efficientnet_b0(pretrained=True)

# Unfreeze last 2 blocks for fine-tuning
for name, param in model.named_parameters():
    if "blocks.5" in name or "blocks.6" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1)

# ============================
# Training Loop
# ============================
best_val_loss = float("inf")
epochs = 300

print("🚀 Training Started...")
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Mixup
        imgs, y_a, y_b, lam = mixup(imgs, labels)

        optimizer.zero_grad()
        outputs = model(imgs)

        # Mixup Loss
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # ============================
    # Validation
    # ============================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            val_loss += criterion(out, labels).item()

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

    # Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': full_dataset.classes
        }, model_path)
        print("💾 Best model saved ✔")

print("🎉 Training Finished.")
print("📌 BEST MODEL:", model_path)