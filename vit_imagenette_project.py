# -*- coding: utf-8 -*-
"""ViT_ImageNette_Project.ipynb

Google Colab Notebook
Vision Transformer (ViT) - 3 Training Variants on ImageNette
AI Class Final Project | Winter 2026 | By: JLei
"""

# ================================================================================
# CHUNK 1 - GPU CHECK
# ================================================================================

import torch
print("=" * 80)
print("GPU CHECK")
print("=" * 80)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("NO GPU - Enable it: Runtime > Change runtime type > GPU")
print("=" * 80)

# ================================================================================
# CHUNK 2 - INSTALL PACKAGES
# ================================================================================

# !pip install --quiet torch torchvision transformers matplotlib seaborn scikit-learn pandas openpyxl tqdm

# print("Packages installed!")

# ================================================================================
# CHUNK 3 - MOUNT DRIVE & CREATE FOLDERS
# ================================================================================

import os

# Mount Google Drive for checkpoints (prevents loss on disconnect)
from google.colab import drive
drive.mount('/content/drive')

# Project folders
PROJECT_DIR = '/content/drive/MyDrive/ViT_Project'
DATA_DIR = './data'
CHECKPOINT_DIR = f'{PROJECT_DIR}/checkpoints'
RESULTS_DIR = f'{PROJECT_DIR}/results'
FIGURES_DIR = f'{PROJECT_DIR}/figures'

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Folders created. Checkpoints save to Google Drive: {CHECKPOINT_DIR}")

# ================================================================================
# CHUNK 4 - CONFIG
# ================================================================================

import warnings
warnings.filterwarnings("ignore")

# === DATASET ===
DATASET_NAME = "ImageNette"
NUM_CLASSES = 10
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# === MODEL ===
PRETRAINED_MODEL = "google/vit-base-patch16-224"

# === TRAINING CONFIGS (per variant) ===
CONFIGS = {
    "from_scratch": {
        "name": "ViT From Scratch",
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 64,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "dropout": 0.1,
    },
    "linear_probe": {
        "name": "ViT Linear Probe",
        "epochs": 20,
        "lr": 1e-2,
        "batch_size": 128,
        "weight_decay": 0.01,
        "warmup_epochs": 0,
        "label_smoothing": 0.0,
        "dropout": 0.0,
    },
    "finetuned": {
        "name": "ViT Fine-tuned",
        "epochs": 10,
        "lr": 2e-5,
        "batch_size": 32,
        "weight_decay": 0.01,
        "warmup_epochs": 2,
        "label_smoothing": 0.1,
        "dropout": 0.0,
    },
}

# === GENERAL ===
RANDOM_SEED = 42
DEVICE = "cuda"
DPI = 150
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = ["#3498db", "#e74c3c", "#2ecc71"]

# === IMAGENETTE CLASS NAMES ===
SYNSET_TO_NAME = {
    'n01440764': 'tench',
    'n02102040': 'springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute',
}
# ImageFolder sorts alphabetically by folder name
CLASS_NAMES = [SYNSET_TO_NAME[k] for k in sorted(SYNSET_TO_NAME.keys())]

print("Config loaded!")
print(f"  Dataset: {DATASET_NAME} ({NUM_CLASSES} classes)")
print(f"  Pretrained: {PRETRAINED_MODEL}")
print(f"  Variants: {', '.join(c['name'] for c in CONFIGS.values())}")

# ================================================================================
# CHUNK 5 - UTILITY FUNCTIONS
# ================================================================================

import math
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


class Logger:
    @staticmethod
    def info(msg): print(f"\033[92m[ INFO ] {msg}\033[0m")
    @staticmethod
    def warning(msg): print(f"\033[93m[ WARN ] {msg}\033[0m")
    @staticmethod
    def error(msg): print(f"\033[91m[ ERROR ] {msg}\033[0m")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Cosine annealing with linear warmup (epoch-level)"""
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


set_seed(RANDOM_SEED)
device = get_device()
Logger.info(f"Device: {device}")
print("Utility functions ready!")

# ================================================================================
# CHUNK 6 - DOWNLOAD & LOAD IMAGENETTE
# ================================================================================

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print("Downloading ImageNette...")

# Download ImageNette2-320 (10 easy ImageNet classes, ~1.5GB)
!wget -q --show-progress https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
!tar -xzf imagenette2-320.tgz -C {DATA_DIR}

TRAIN_DIR = f'{DATA_DIR}/imagenette2-320/train'
VAL_DIR = f'{DATA_DIR}/imagenette2-320/val'

# Data transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize(256),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=val_transform)

Logger.info(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
Logger.info(f"Classes: {CLASS_NAMES}")

# Show sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
mean_t = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
std_t = torch.tensor(NORMALIZE_STD).view(3, 1, 1)

indices = np.random.choice(len(train_dataset), 10, replace=False)
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[indices[i]]
    img_display = torch.clamp(img * std_t + mean_t, 0, 1)
    ax.imshow(img_display.permute(1, 2, 0).numpy())
    ax.set_title(CLASS_NAMES[label], fontsize=10)
    ax.axis('off')

plt.suptitle('ImageNette Sample Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/sample_images.png", dpi=DPI, bbox_inches='tight')
plt.show()
print("Data ready!")

# ================================================================================
# CHUNK 7 - MODEL DEFINITIONS (3 ViT VARIANTS)
# ================================================================================

from transformers import ViTForImageClassification, ViTConfig

print("Setting up ViT model variants...")


def create_vit_from_scratch():
    """ViT-Base with random initialization (no pretrained weights)"""
    config = ViTConfig(
        image_size=IMAGE_SIZE,
        patch_size=16,
        num_channels=3,
        num_labels=NUM_CLASSES,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = ViTForImageClassification(config)
    return model


def create_vit_linear_probe():
    """Pretrained ViT-Base with frozen backbone (only classifier trains)"""
    model = ViTForImageClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    # Freeze the entire ViT backbone
    for param in model.vit.parameters():
        param.requires_grad = False
    return model


def create_vit_finetuned():
    """Pretrained ViT-Base with all parameters trainable"""
    model = ViTForImageClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model


# Model factory
MODEL_CREATORS = {
    "from_scratch": create_vit_from_scratch,
    "linear_probe": create_vit_linear_probe,
    "finetuned": create_vit_finetuned,
}

# Quick check
for variant_key, creator in MODEL_CREATORS.items():
    model_tmp = creator()
    total_p = count_parameters(model_tmp, trainable_only=False)
    train_p = count_parameters(model_tmp, trainable_only=True)
    Logger.info(f"{CONFIGS[variant_key]['name']}: "
                f"{total_p:,} total params | {train_p:,} trainable")
    del model_tmp

print("Model definitions ready!")

# ================================================================================
# CHUNK 8 - TRAINING & EVALUATION FUNCTIONS
# ================================================================================

import time
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm


def train_epoch(model, loader, optimizer, device, epoch, num_epochs, variant_name,
                label_smoothing=0.0):
    """Train for one epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}/{num_epochs} [{variant_name}] Train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        # Gradient clipping (important for from-scratch training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    return running_loss / len(loader), 100. * correct / total


def evaluate_epoch(model, loader, device):
    """Evaluate on validation set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def train_variant(variant_key):
    """Full training pipeline for one ViT variant"""
    cfg = CONFIGS[variant_key]
    name = cfg['name']

    print(f"\n{'=' * 80}")
    Logger.info(f"TRAINING: {name}")
    print(f"{'=' * 80}")
    print(f"  Epochs: {cfg['epochs']} | LR: {cfg['lr']} | Batch: {cfg['batch_size']}")
    print(f"  Warmup: {cfg['warmup_epochs']} epochs | Weight Decay: {cfg['weight_decay']}")
    print(f"{'=' * 80}\n")

    # Create model
    model = MODEL_CREATORS[variant_key]().to(device)
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    Logger.info(f"Params: {trainable:,} trainable / {total:,} total")

    # Create data loaders with variant-specific batch size
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    # Optimizer & scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = get_warmup_cosine_scheduler(
        optimizer, cfg['warmup_epochs'], cfg['epochs'])

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    start_time = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, cfg['epochs'],
            name, label_smoothing=cfg['label_smoothing'])
        val_loss, val_acc = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        Logger.info(f"Epoch {epoch}/{cfg['epochs']} - "
                    f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f"{CHECKPOINT_DIR}/{variant_key}_best.pth")
            Logger.info(f"New best: {best_val_acc:.2f}%")

        # Periodic checkpoint (every 5 epochs)
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       f"{CHECKPOINT_DIR}/{variant_key}_epoch{epoch}.pth")

    elapsed = (time.time() - start_time) / 60
    Logger.info(f"{name} done in {elapsed:.1f} min | Best val acc: {best_val_acc:.2f}%")

    # Reload best model for evaluation
    model.load_state_dict(
        torch.load(f"{CHECKPOINT_DIR}/{variant_key}_best.pth", weights_only=True))

    return model, history


def evaluate_full(model, loader, device):
    """Full evaluation with all metrics"""
    model.eval()
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images)
            probs = torch.softmax(outputs.logits, 1)
            _, preds = torch.max(outputs.logits, 1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    targets = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    acc = 100. * accuracy_score(targets, preds_arr)
    metrics = calculate_metrics(targets, preds_arr, probs_arr)

    return {
        'test_acc': acc,
        **metrics,
        'targets': targets,
        'predictions': preds_arr,
        'probabilities': probs_arr,
    }


print("Training functions ready!")

# ================================================================================
# CHUNK 9 - TRAIN VIT FROM SCRATCH
# ================================================================================
# Expected: ~40-60% accuracy (ViT is data-hungry without pretraining)
# Time: ~25-40 min on Colab T4

print("=" * 80)
print("EXPERIMENT 1: ViT From Scratch")
print("=" * 80)
print("Training ViT-Base with random initialization...")
print("This WILL perform poorly - that's the point!\n")

model_scratch, history_scratch = train_variant("from_scratch")

# ================================================================================
# CHUNK 10 - TRAIN VIT LINEAR PROBE
# ================================================================================
# Expected: ~75-85% accuracy (pretrained features, frozen backbone)
# Time: ~3-5 min on Colab T4

print("=" * 80)
print("EXPERIMENT 2: ViT Linear Probe")
print("=" * 80)
print("Pretrained backbone FROZEN - only training the classification head...\n")

model_probe, history_probe = train_variant("linear_probe")

# ================================================================================
# CHUNK 11 - TRAIN VIT FINE-TUNED
# ================================================================================
# Expected: ~92-97% accuracy (full transfer learning)
# Time: ~8-12 min on Colab T4

print("=" * 80)
print("EXPERIMENT 3: ViT Fine-tuned")
print("=" * 80)
print("Pretrained backbone + fine-tuning ALL layers...\n")

model_finetuned, history_finetuned = train_variant("finetuned")

print("\n" + "=" * 80)
Logger.info("ALL 3 EXPERIMENTS COMPLETE!")
print("=" * 80)

# ================================================================================
# CHUNK 12 - EVALUATE ALL MODELS ON TEST SET
# ================================================================================

print("Evaluating all models on validation set...")

# Create val loader with fixed batch size
eval_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

all_models = {
    "from_scratch": model_scratch,
    "linear_probe": model_probe,
    "finetuned": model_finetuned,
}
all_histories = {
    "from_scratch": history_scratch,
    "linear_probe": history_probe,
    "finetuned": history_finetuned,
}
all_results = {}

for variant_key, model in all_models.items():
    name = CONFIGS[variant_key]['name']
    Logger.info(f"Evaluating {name}...")
    result = evaluate_full(model, eval_loader, device)
    all_results[variant_key] = result

    Logger.info(f"  Accuracy: {result['test_acc']:.2f}%")
    Logger.info(f"  Precision: {result['precision']:.4f} | "
                f"Recall: {result['recall']:.4f}")
    Logger.info(f"  F1: {result['f1_score']:.4f} | "
                f"ROC-AUC: {result['roc_auc']:.4f}\n")

print("Evaluation done!")

# ================================================================================
# CHUNK 13 - VISUALIZATIONS
# ================================================================================

print("Creating visualizations...")

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

sns.set_palette(COLOR_PALETTE)

# --- 1. Training Curves ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for idx, (variant_key, hist) in enumerate(all_histories.items()):
    name = CONFIGS[variant_key]['name']
    color = COLOR_PALETTE[idx]
    epochs = range(1, len(hist['train_loss']) + 1)

    axes[0, 0].plot(epochs, hist['train_loss'], label=name, color=color, linewidth=2)
    axes[0, 1].plot(epochs, hist['val_loss'], label=name, color=color, linewidth=2)
    axes[1, 0].plot(epochs, hist['train_acc'], label=name, color=color, linewidth=2)
    axes[1, 1].plot(epochs, hist['val_acc'], label=name, color=color, linewidth=2)

titles = ['Training Loss', 'Validation Loss', 'Training Accuracy (%)',
          'Validation Accuracy (%)']
for ax, title in zip(axes.flat, titles):
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch')

plt.suptitle('Training Curves - 3 ViT Variants', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/training_curves.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("training_curves.png saved!")

# --- 2. Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for idx, (variant_key, result) in enumerate(all_results.items()):
    name = CONFIGS[variant_key]['name']
    cm = confusion_matrix(result['targets'], result['predictions'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{name}\nAcc: {result['test_acc']:.2f}%", fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].tick_params(axis='y', rotation=0)

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/confusion_matrices.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("confusion_matrices.png saved!")

# --- 3. Per-Class Accuracy ---
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(NUM_CLASSES)
width = 0.25

for idx, (variant_key, result) in enumerate(all_results.items()):
    name = CONFIGS[variant_key]['name']
    cm = confusion_matrix(result['targets'], result['predictions'])
    per_class_acc = [100 * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                     for i in range(NUM_CLASSES)]

    offset = width * (idx - 1)
    bars = ax.bar(x + offset, per_class_acc, width, label=name,
                  color=COLOR_PALETTE[idx], alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Accuracy Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 110])

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/per_class_accuracy.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("per_class_accuracy.png saved!")

# --- 4. Metrics Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 5))
metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_names))
width = 0.25

for idx, (variant_key, result) in enumerate(all_results.items()):
    name = CONFIGS[variant_key]['name']
    values = [result[m] for m in metrics_names]
    offset = width * (idx - 1)
    bars = ax.bar(x + offset, values, width, label=name,
                  color=COLOR_PALETTE[idx])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Metrics', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim([0, 1.15])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/metrics_comparison.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("metrics_comparison.png saved!")

# --- 5. ROC Curves ---
fig, ax = plt.subplots(figsize=(10, 8))

for idx, (variant_key, result) in enumerate(all_results.items()):
    name = CONFIGS[variant_key]['name']
    y_true = label_binarize(result['targets'], classes=range(NUM_CLASSES))
    y_score = result['probabilities']

    fpr_list, tpr_list = [], []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    ax.plot(all_fpr, mean_tpr,
            label=f"{name} (AUC = {macro_auc:.4f})",
            color=COLOR_PALETTE[idx], linewidth=3, alpha=0.8)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2,
        label='Random (AUC = 0.5000)', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('ROC Curves - Macro-Average (One-vs-Rest)',
             fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/roc_curves.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("roc_curves.png saved!")

# --- 6. Parameter Efficiency ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

variant_names = [CONFIGS[k]['name'] for k in all_results.keys()]
total_params = []
trainable_params = []
accuracies = []

for variant_key in all_results.keys():
    model_tmp = MODEL_CREATORS[variant_key]()
    total_params.append(count_parameters(model_tmp, trainable_only=False) / 1e6)
    trainable_params.append(count_parameters(model_tmp, trainable_only=True) / 1e6)
    accuracies.append(all_results[variant_key]['test_acc'])
    del model_tmp

# Plot 1: Trainable params comparison
ax = axes[0]
x_pos = np.arange(len(variant_names))
bars = ax.bar(x_pos, trainable_params, color=COLOR_PALETTE, alpha=0.8,
              edgecolor='black', linewidth=1.5)
for bar, acc in zip(bars, accuracies):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
            f'{acc:.1f}% acc', ha='center', va='bottom', fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_ylabel('Trainable Parameters (M)', fontsize=12, fontweight='bold')
ax.set_title('Trainable Parameters per Variant', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(variant_names, rotation=15, ha='right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Efficiency scatter
ax = axes[1]
scatter = ax.scatter(trainable_params, accuracies, s=300, c=COLOR_PALETTE,
                     alpha=0.7, edgecolors='black', linewidth=2)
for name, tp, acc in zip(variant_names, trainable_params, accuracies):
    ax.annotate(name, (tp, acc), xytext=(10, 10), textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

ax.set_xlabel('Trainable Parameters (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Parameter Efficiency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/parameter_efficiency.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("parameter_efficiency.png saved!")

print("\nAll standard visualizations complete!")

# ================================================================================
# CHUNK 14 - VIT-SPECIFIC VISUALIZATIONS (ATTENTION MAPS)
# ================================================================================

print("Creating ViT-specific visualizations...")

# --- 1. Attention Map Heatmaps (using fine-tuned model) ---
print("Generating attention map overlays...")

model_finetuned.config.output_attentions = True
model_finetuned.eval()

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
# Row 0: Original images
# Row 1: Attention heatmap
# Row 2: Overlay (original + attention)

sample_indices = np.random.choice(len(val_dataset), 6, replace=False)

for col, img_idx in enumerate(sample_indices):
    image, label = val_dataset[img_idx]

    # Unnormalize for display
    display_img = torch.clamp(image * std_t + mean_t, 0, 1)
    display_np = display_img.permute(1, 2, 0).numpy()

    # Get attention weights
    with torch.no_grad():
        outputs = model_finetuned(
            pixel_values=image.unsqueeze(0).to(device),
            output_attentions=True)

    # Last layer attention, average across heads
    attn = outputs.attentions[-1].squeeze(0)  # (12, 197, 197)
    attn = attn.mean(0)  # (197, 197)
    cls_attn = attn[0, 1:]  # (196,) CLS token -> all patches
    cls_attn = cls_attn.reshape(1, 1, 14, 14).float()

    # Upsample to image size
    cls_attn_up = F.interpolate(cls_attn, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
    cls_attn_np = cls_attn_up.squeeze().cpu().numpy()
    # Normalize to [0, 1]
    cls_attn_np = (cls_attn_np - cls_attn_np.min()) / \
                  (cls_attn_np.max() - cls_attn_np.min() + 1e-8)

    # Row 0: Original
    axes[0, col].imshow(display_np)
    axes[0, col].set_title(CLASS_NAMES[label], fontsize=11, fontweight='bold')
    axes[0, col].axis('off')

    # Row 1: Attention heatmap only
    axes[1, col].imshow(cls_attn_np, cmap='inferno')
    axes[1, col].axis('off')

    # Row 2: Overlay
    axes[2, col].imshow(display_np)
    axes[2, col].imshow(cls_attn_np, cmap='jet', alpha=0.5)
    axes[2, col].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold', rotation=90,
                       labelpad=15)
axes[1, 0].set_ylabel('Attention', fontsize=12, fontweight='bold', rotation=90,
                       labelpad=15)
axes[2, 0].set_ylabel('Overlay', fontsize=12, fontweight='bold', rotation=90,
                       labelpad=15)

plt.suptitle('ViT Attention Maps (Fine-tuned Model, Last Layer)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/attention_maps.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("attention_maps.png saved!")

# --- 2. Multi-Layer Attention (how attention evolves through layers) ---
print("Generating multi-layer attention comparison...")

sample_img, sample_label = val_dataset[sample_indices[0]]

with torch.no_grad():
    outputs = model_finetuned(
        pixel_values=sample_img.unsqueeze(0).to(device),
        output_attentions=True)

# Show attention from layers 1, 4, 8, 12
layer_indices = [0, 3, 7, 11]
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

# Original image
display_img = torch.clamp(sample_img * std_t + mean_t, 0, 1)
axes[0].imshow(display_img.permute(1, 2, 0).numpy())
axes[0].set_title(f'Original\n({CLASS_NAMES[sample_label]})', fontsize=10)
axes[0].axis('off')

for i, layer_idx in enumerate(layer_indices):
    attn = outputs.attentions[layer_idx].squeeze(0).mean(0)  # Avg across heads
    cls_attn = attn[0, 1:].reshape(1, 1, 14, 14).float()
    cls_attn_up = F.interpolate(cls_attn, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
    cls_attn_np = cls_attn_up.squeeze().cpu().numpy()
    cls_attn_np = (cls_attn_np - cls_attn_np.min()) / \
                  (cls_attn_np.max() - cls_attn_np.min() + 1e-8)

    axes[i + 1].imshow(display_img.permute(1, 2, 0).numpy())
    axes[i + 1].imshow(cls_attn_np, cmap='jet', alpha=0.5)
    axes[i + 1].set_title(f'Layer {layer_idx + 1}', fontsize=10)
    axes[i + 1].axis('off')

plt.suptitle('Attention Evolution Across Transformer Layers',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/attention_layers.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("attention_layers.png saved!")

# --- 3. Position Embedding Similarity ---
print("Generating position embedding visualization...")

pos_embed = model_finetuned.vit.embeddings.position_embeddings.squeeze(0)
patch_embed = pos_embed[1:]  # Remove CLS token: (196, 768)
patch_embed_norm = F.normalize(patch_embed, dim=-1)
similarity = torch.mm(patch_embed_norm, patch_embed_norm.t())  # (196, 196)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# Reference positions: corners, edges, center
ref_positions = [0, 6, 13, 91, 97, 104, 182, 195]
ref_labels = ['(0,0)', '(0,6)', '(0,13)', '(6,7)',
              '(6,13)', '(7,6)', '(13,0)', '(13,13)']

for idx, (pos, label) in enumerate(zip(ref_positions, ref_labels)):
    row, col = idx // 4, idx % 4
    sim = similarity[pos].reshape(14, 14).cpu().detach().numpy()
    im = axes[row, col].imshow(sim, cmap='viridis')
    ref_r, ref_c = pos // 14, pos % 14
    axes[row, col].scatter([ref_c], [ref_r], c='red', s=100, marker='x',
                           linewidths=2)
    axes[row, col].set_title(f'Ref pos {label}', fontsize=10)
    axes[row, col].axis('off')

plt.suptitle('Position Embedding Cosine Similarity\n'
             '(Red X = reference, brighter = more similar)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/position_embeddings.png", dpi=DPI, bbox_inches='tight')
plt.show()
Logger.info("position_embeddings.png saved!")

print("\nAll ViT-specific visualizations complete!")

# ================================================================================
# CHUNK 15 - RESULTS SUMMARY & DOWNLOAD
# ================================================================================

import pandas as pd

# --- Training Summary ---
train_data = []
for variant_key, hist in all_histories.items():
    cfg = CONFIGS[variant_key]
    model_tmp = MODEL_CREATORS[variant_key]()
    train_data.append({
        'Variant': cfg['name'],
        'Epochs': cfg['epochs'],
        'Best Val Acc (%)': f"{max(hist['val_acc']):.2f}",
        'Final Train Acc (%)': f"{hist['train_acc'][-1]:.2f}",
        'Final Val Acc (%)': f"{hist['val_acc'][-1]:.2f}",
        'Trainable Params (M)': f"{count_parameters(model_tmp, True) / 1e6:.1f}",
        'Total Params (M)': f"{count_parameters(model_tmp, False) / 1e6:.1f}",
        'LR': cfg['lr'],
        'Batch Size': cfg['batch_size'],
    })
    del model_tmp
train_df = pd.DataFrame(train_data)

# --- Test Metrics ---
test_data = []
for variant_key, result in all_results.items():
    cfg = CONFIGS[variant_key]
    test_data.append({
        'Variant': cfg['name'],
        'Accuracy (%)': f"{result['test_acc']:.2f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1-Score': f"{result['f1_score']:.4f}",
        'ROC-AUC': f"{result['roc_auc']:.4f}",
    })
test_df = pd.DataFrame(test_data)

# Display tables
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(train_df.to_string(index=False))
print("\n" + "=" * 70)
print("TEST METRICS")
print("=" * 70)
print(test_df.to_string(index=False))
print("=" * 70)

# --- Save as CSV ---
train_df.to_csv(f"{RESULTS_DIR}/training_summary.csv", index=False)
test_df.to_csv(f"{RESULTS_DIR}/test_metrics.csv", index=False)
Logger.info(f"CSVs saved to {RESULTS_DIR}/")

# --- Download ---
print(f"\nCreating download package...")
!cd "{PROJECT_DIR}" && zip -r /content/ViT_Project_Complete.zip checkpoints/ figures/ results/ -q

from google.colab import files
print(f"\nClick below to download:")
files.download('/content/ViT_Project_Complete.zip')

# --- Final Summary ---
print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)

best_variant = max(all_results.items(), key=lambda x: x[1]['test_acc'])
best_key, best_result = best_variant

print(f"\nBEST: {CONFIGS[best_key]['name']}")
print(f"   Accuracy: {best_result['test_acc']:.2f}%")
print(f"   F1-Score: {best_result['f1_score']:.4f}")
print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")

print(f"\nALL RESULTS:")
for variant_key in sorted(all_results.keys(),
                          key=lambda k: all_results[k]['test_acc'], reverse=True):
    name = CONFIGS[variant_key]['name']
    acc = all_results[variant_key]['test_acc']
    print(f"   {name:25s}: {acc:6.2f}%")

print(f"\nFIGURES GENERATED:")
print(f"   1. sample_images.png")
print(f"   2. training_curves.png")
print(f"   3. confusion_matrices.png")
print(f"   4. per_class_accuracy.png")
print(f"   5. metrics_comparison.png")
print(f"   6. roc_curves.png")
print(f"   7. parameter_efficiency.png")
print(f"   8. attention_maps.png")
print(f"   9. attention_layers.png")
print(f"  10. position_embeddings.png")

print(f"\nKEY TAKEAWAY:")
print(f"   ViT needs pretraining! From scratch vs fine-tuned shows")
print(f"   the massive impact of transfer learning on small datasets.")
print("=" * 80)
