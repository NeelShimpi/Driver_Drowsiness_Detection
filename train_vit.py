import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# --- HARDWARE COMPATIBILITY FIX FOR RTX 50-SERIES ---
# Disable pre-compiled fused attention kernels that crash on sm_120
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # The ultimate sledgehammer for RTX 50-series: disable cuDNN convolutions
    torch.backends.cudnn.enabled = False
# ----------------------------------------------------

# =========================
# CONFIG
# =========================

DATASET_DIR = r"C:\Users\neels\MajorProject\project\driver_drowsiness_dataset\dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# DATA TRANSFORMS
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# LOAD DATASET
# =========================

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

class_names = dataset.classes
print("Classes:", class_names)

# =========================
# SPLIT DATASET
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Training samples:", train_size)
print("Validation samples:", val_size)

# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL
# =========================

import timm

# We use vit_tiny_patch16_224 since it fits beautifully on a laptop for real-time video streaming
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=len(class_names))

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# TRAINING LOOP
# =========================

train_acc_history = []
val_acc_history = []

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # TRAIN
    model.train()

    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_acc_history.append(train_acc)

    print("Train Loss:", running_loss/len(train_loader))
    print("Train Accuracy:", train_acc)

    # VALIDATION
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_acc_history.append(val_acc)

    print("Validation Accuracy:", val_acc)

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "drowsiness_model_timm.pth")

print("\nModel saved!")

# =========================
# TRAINING CURVE
# =========================

plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Curve")

plt.legend()

#plt.savefig("training_curve.png")

print("Training curve saved!")