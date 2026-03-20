# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI class final project (Winter 2026, Dr. Muheidat) comparing 3 Vision Transformer training strategies on ImageNette. Builds on previous ML class work (CNN vs ResNet on CIFAR-10). This project expands from *architecture comparison* (CNN vs ResNet) to *training strategy comparison* (scratch vs linear probe vs fine-tuned) using a fundamentally different paradigm (self-attention vs convolutions).

## Key Files

- `ViT_ImageNette_Project.ipynb` — Main deliverable (Colab notebook, 16 cells)
- `vit_imagenette_project.py` — Same code as flat Python script (Colab export format)
- `latex/presentation.tex` — Beamer presentation (Warsaw theme, 23 slides)
- `latex/paper.tex` — Final report (5 pages, article class)
- `latex/figures/` — ViT experiment output figures (10 PNGs)
- `latex/prev_figures/` — Previous CNN/ResNet project figures (for intro slides)
- `planning_summary.md` — Project planning decisions and rationale
- `session/` — Session logs documenting Claude Code conversations
- `Previous Project/` — Last semester's ML class code (reference only, not runnable)

## Architecture

Single notebook structured in 15 sequential code chunks + 1 markdown header. Three ViT-Base experiments using `google/vit-base-patch16-224`:
- **From Scratch**: Random init, all params trainable (50 epochs, LR 1e-3, batch 64) — **37.30% accuracy**
- **Linear Probe**: Pretrained backbone frozen, only classification head trains (20 epochs, LR 1e-2, batch 128) — **99.67% accuracy (winner)**
- **Fine-tuned**: Pretrained backbone, all layers trainable (10 epochs, LR 2e-5, batch 32) — **99.59% accuracy**

Dataset: ImageNette (10 ImageNet classes, ~9.5k train / ~3.9k val, 224x224 images). Not CIFAR-10 — ViT needs full-resolution images.

All three variants use AdamW optimizer with cosine annealing + linear warmup scheduler. HuggingFace ViT models expect `pixel_values` keyword arg and normalization with mean/std = [0.5, 0.5, 0.5].

## Known Issue: Attention Maps

When generating attention maps (chunk 14), the model config must be set to eager attention before enabling output_attentions:
```python
model_finetuned.config._attn_implementation = "eager"
model_finetuned.config.output_attentions = True
```

## Key Dependencies

PyTorch, HuggingFace `transformers` (ViTForImageClassification, ViTConfig), torchvision, scikit-learn, matplotlib, seaborn, pandas.

## Running

This code runs on **Google Colab with GPU**, not locally. It uses Colab-specific features (`drive.mount`, `!wget`, `files.download`). Upload the `.ipynb` to Colab and run cells sequentially. Checkpoints save to Google Drive (`/content/drive/MyDrive/ViT_Project/`).

## LaTeX Compilation

```bash
cd latex && pdflatex -interaction=nonstopmode presentation.tex
cd latex && pdflatex -interaction=nonstopmode paper.tex
```

The presentation uses Warsaw theme with Boadilla/CambridgeUS/Madrid layering + infolines outer theme + bookman font. This is Junjie's standard Beamer template — do not change the theme stack.

## Editing Workflow

When fixing bugs or updating code, edit **both** files to keep them in sync:
- `vit_imagenette_project.py` — edit with Edit tool
- `ViT_ImageNette_Project.ipynb` — edit with NotebookEdit tool (cell IDs are `cell-0` through `cell-15`)

## Previous Project (Reference Only)

`Previous Project/` contains last semester's ML class work (CNN vs ResNet on CIFAR-10). Key differences:
- Previous: architecture comparison (CNN vs ResNet-15 vs ResNet-30), CIFAR-10 (32x32), SGD optimizer, best 94.28%
- Current: training strategy comparison (scratch vs probe vs fine-tune), ImageNette (224x224), AdamW optimizer, best 99.67%

## Output Preferences

- Never export as Excel (.xlsx), Word (.docx), or Microsoft formats unless explicitly asked
- CSV or console output for results
- PNG figures saved to Google Drive `figures/` directory
