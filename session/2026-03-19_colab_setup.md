# Session: 2026-03-19 — Colab Setup & Project Framing

## What We Did

### 1. Notebook Preparation
- Reviewed the full `vit_imagenette_project.py` (15 chunks, 1051 lines)
- Converted `.py` script to proper `.ipynb` notebook (`ViT_ImageNette_Project.ipynb`) with 16 cells (1 markdown header + 15 code cells)
- Set Colab metadata for T4 GPU

### 2. Bug Fix
- **`total_mem` → `total_memory`**: `torch.cuda.get_device_properties(0).total_mem` is not a valid attribute. Fixed to `total_memory` in both `.py` and `.ipynb` files.

### 3. GPU Selection
- Evaluated Colab GPU options (CPU, T4, L4, A100, H100, TPUs)
- **Chose T4**: Free tier, 16GB VRAM, sufficient for ViT-Base, code designed for it
- TPUs won't work (code uses CUDA, would need XLA rewrite)

### 4. Colab Testing Progress
- Chunk 1 (GPU check): Fixed and working
- Chunk 2 (packages): Working
- Chunk 3 (Drive mount): Working
- Chunk 4 (config): Working
- Chunk 5 (utilities): Working
- Chunk 6 (data download): Working
- Chunk 7 (model definitions): Working — all 3 variants load correctly
  - From Scratch: 85.8M total / 85.8M trainable
  - Linear Probe: 85.8M total / 7,690 trainable
  - Fine-tuned: 85.8M total / 85.8M trainable
- HuggingFace warnings (unauthenticated, classifier size mismatch) are expected and harmless
- Training chunks (9-11) not yet run as of this session

### 5. Previous Project Comparison
- Reviewed previous CNN/ResNet project from GitHub (`MSCS-CNN-ResNET15-30`)
- Read local copy at `Previous Project/cnn&resnet_on_cifar_10.py`
- Identified features in previous project missing from current: convergence plots, hyperparameter sensitivity analysis, data augmentation impact, shaded ROC curves
- Decision: May add these later, not critical for initial run

### 6. Project Narrative (for Report/Presentation)
Framed the expansion story:
- **Previous**: Architecture comparison (CNN vs ResNet) on CIFAR-10 — "Does depth help?"
- **Current**: Training strategy comparison (scratch vs probe vs fine-tune) on ImageNette — "Does pretraining matter?"
- Key shift: From comparing *model structures* to comparing *training strategies* using Vision Transformers (self-attention paradigm)

### 7. CLAUDE.md Update
- Added `.ipynb` as main deliverable
- Added editing workflow (both files need syncing)
- Added project narrative and previous/current comparison

## Next Steps
- Run training chunks 9-11 on Colab (~40-60 min total)
- Run evaluation and visualization chunks 12-15
- Consider adding convergence/hyperparameter analysis chunks from previous project
- Write LaTeX presentation
- Write final report

## Title Chosen
"ViT Training Strategies on ImageNette: From Scratch vs. Linear Probe vs. Fine-Tuning"
