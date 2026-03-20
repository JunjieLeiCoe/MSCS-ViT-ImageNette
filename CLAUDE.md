# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI class final project (Winter 2026, Dr. Muheidat) comparing 3 Vision Transformer training strategies on ImageNette. Builds on previous ML class work (CNN vs ResNet on CIFAR-10). This project expands from *architecture comparison* (CNN vs ResNet) to *training strategy comparison* (scratch vs linear probe vs fine-tuned) using a fundamentally different paradigm (self-attention vs convolutions).

## Key Files

- `ViT_ImageNette_Project.ipynb` — Main deliverable (Colab notebook, 16 cells)
- `vit_imagenette_project.py` — Same code as flat Python script (Colab export format)
- `planning_summary.md` — Project planning decisions and rationale
- `Previous Project/` — Last semester's ML class code (reference only, not runnable)

## Architecture

Single notebook structured in 15 sequential code chunks + 1 markdown header. Three ViT-Base experiments using `google/vit-base-patch16-224`:
- **From Scratch**: Random init, all params trainable (50 epochs, LR 1e-3, batch 64)
- **Linear Probe**: Pretrained backbone frozen, only classification head trains (20 epochs, LR 1e-2, batch 128)
- **Fine-tuned**: Pretrained backbone, all layers trainable (10 epochs, LR 2e-5, batch 32)

Dataset: ImageNette (10 ImageNet classes, ~9.5k train / ~3.9k val, 224x224 images). Not CIFAR-10 — ViT needs full-resolution images.

All three variants use AdamW optimizer with cosine annealing + linear warmup scheduler. HuggingFace ViT models expect `pixel_values` keyword arg and normalization with mean/std = [0.5, 0.5, 0.5].

## Key Dependencies

PyTorch, HuggingFace `transformers` (ViTForImageClassification, ViTConfig), torchvision, scikit-learn, matplotlib, seaborn, pandas.

## Running

This code runs on **Google Colab with T4 GPU**, not locally. It uses Colab-specific features (`drive.mount`, `!wget`, `files.download`). Upload the `.ipynb` to Colab and run cells sequentially. Checkpoints save to Google Drive (`/content/drive/MyDrive/ViT_Project/`). Total GPU training time: ~40-60 min on T4.

## Colab Resilience

Checkpoints save to Google Drive every 5 epochs + on best validation accuracy. If Colab disconnects, re-run from the last checkpoint rather than restarting from scratch.

## Editing Workflow

When fixing bugs or updating code, edit **both** files to keep them in sync:
- `vit_imagenette_project.py` — edit with Edit tool
- `ViT_ImageNette_Project.ipynb` — edit with NotebookEdit tool (cell IDs are `cell-0` through `cell-15`)

## Previous Project (Reference Only)

`Previous Project/` contains last semester's ML class work (CNN vs ResNet on CIFAR-10). Key differences from current project:
- Previous: architecture comparison (CNN vs ResNet-15 vs ResNet-30), CIFAR-10 (32x32), SGD optimizer
- Current: training strategy comparison (scratch vs probe vs fine-tune), ImageNette (224x224), AdamW optimizer, ViT-specific visualizations (attention maps, position embeddings)

## Output Preferences

- Never export as Excel (.xlsx), Word (.docx), or Microsoft formats unless explicitly asked
- CSV or console output for results
- PNG figures saved to Google Drive `figures/` directory
