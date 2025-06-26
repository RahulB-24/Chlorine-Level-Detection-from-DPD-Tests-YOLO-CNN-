# train_cnn.py
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import r2_score, mean_absolute_error
from collections import defaultdict

# ----------------- Config ----------------- #
class Config:
    DATA_DIR    = 'data/cropped_without_invalid'
    TRAIN_DIR   = os.path.join(DATA_DIR, 'train')
    VAL_DIR     = os.path.join(DATA_DIR, 'val')
    MODEL_PATH  = 'models/cnn_chlorine_best3.pth'
    IMG_SIZE    = 224
    BATCH_SIZE  = 32
    EPOCHS      = 50
    LR          = 1e-4
    WEIGHT_DECAY= 1e-5
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED        = 42
    CLIP_NORM   = 1.0
    LR_FACTOR   = 0.5
    LR_PATIENCE = 3
    MIN_LR      = 1e-6

# ----------------- Reproducibility ----------------- #
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.DEVICE=='cuda':
        torch.cuda.manual_seed_all(seed)
set_seed()

# ----------------- Dataset ----------------- #
class ChlorineDataset(Dataset):
    def __init__(self, root_dir, transform=None, normalize_labels=True):
        self.samples = []
        self.transform = transform
        self.normalize_labels = normalize_labels
        self.min_ppm = float('inf')
        self.max_ppm = float('-inf')

        for folder in os.listdir(root_dir):
            try:
                ppm = float(folder)
                self.min_ppm = min(self.min_ppm, ppm)
                self.max_ppm = max(self.max_ppm, ppm)
            except ValueError:
                continue

        for folder in os.listdir(root_dir):
            try:
                ppm = float(folder)
            except ValueError:
                continue
            folder_path = os.path.join(root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', 'jpeg', 'png')):
                    self.samples.append((os.path.join(folder_path, fname), ppm))

        print(f"📦 Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = (label - self.min_ppm) / (self.max_ppm - self.min_ppm) if self.normalize_labels else label
        return img, torch.tensor([label], dtype=torch.float32)

# ----------------- Transforms ----------------- #
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE,Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ----------------- Model ----------------- #
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(Config.DEVICE)

# ----------------- Training Loop ----------------- #
def denormalize(labels, min_val, max_val):
    return labels * (max_val - min_val) + min_val

def train():
    # Datasets & Loaders
    train_ds = ChlorineDataset(Config.TRAIN_DIR, train_transform)
    val_ds   = ChlorineDataset(Config.VAL_DIR, val_transform)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model, optimizer, criterion, scheduler, scaler
    model     = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     factor=Config.LR_FACTOR,
                                                     patience=Config.LR_PATIENCE,
                                                     min_lr=Config.MIN_LR)
    scaler    = amp.GradScaler()
    best_r2   = -float('inf')

    # Load existing best if available
    if os.path.isfile(Config.MODEL_PATH):
        print("📦 Loading previous best model...")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(Config.DEVICE)
                out = model(x).cpu().numpy().flatten()
                all_preds.extend(out)
                all_labels.extend(y.numpy().flatten())
        denorm_preds  = denormalize(np.array(all_preds), val_ds.min_ppm, val_ds.max_ppm)
        denorm_labels = denormalize(np.array(all_labels), val_ds.min_ppm, val_ds.max_ppm)
        best_r2 = r2_score(denorm_labels, denorm_preds)
        print(f"✅ Previous best R² = {best_r2:.4f}")

    # Epoch loop
    for epoch in range(1, Config.EPOCHS+1):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", unit='batch'):
            imgs   = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            with amp.autocast():
                preds = model(imgs)
                loss  = criterion(preds, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(Config.DEVICE)
                out  = model(imgs).cpu().numpy().flatten()
                val_preds.extend(out)
                val_labels.extend(labels.numpy().flatten())

        denorm_preds  = denormalize(np.array(val_preds), val_ds.min_ppm, val_ds.max_ppm)
        denorm_labels = denormalize(np.array(val_labels), val_ds.min_ppm, val_ds.max_ppm)

        epoch_r2  = r2_score(denorm_labels, denorm_preds)
        epoch_mae = mean_absolute_error(denorm_labels, denorm_preds)
        epoch_mse = np.mean((denorm_labels - denorm_preds)**2)
        avg_train_loss = running_loss / len(train_loader)

        print(f"📊 Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val R²={epoch_r2:.4f} | "
              f"MAE={epoch_mae:.4f} | MSE={epoch_mse:.4f}")

        # Per-bin evaluation (optional diagnostic)
        bin_results = defaultdict(list)
        for true, pred in zip(denorm_labels, denorm_preds):
            bin_label = round(true, 1)
            bin_results[bin_label].append((true, pred))
        for k in sorted(bin_results.keys()):
            truths, preds = zip(*bin_results[k])
            print(f"    - ppm={k:.1f} | R²={r2_score(truths, preds):.3f} | MAE={mean_absolute_error(truths, preds):.3f}")

        # LR scheduler step on val R²
        scheduler.step(epoch_r2)

        # Save if strictly better
        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"✅ Saved new best model (R² improved to {best_r2:.4f})")

if __name__ == '__main__':
    train()
