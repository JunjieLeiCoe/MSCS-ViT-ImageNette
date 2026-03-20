# ViT Project AI - Planning Summary

## What Was Done in Plan Mode

During the planning phase, we discussed and finalized the project scope, architecture, and timeline for a Vision Transformer (ViT) image classification project.

### Key Decisions Made

1. **Model Choice**: ViT only (3 variants) instead of ViT + AlexNet
   - AlexNet was deemed a step backward given prior CNN/ResNet experience
   - Three ViT variants provide a cleaner, more focused comparison

2. **Three ViT Variants**
   - **From Scratch** - Random init, train all params on ImageNette
   - **Linear Probe** - Pretrained backbone frozen, only train classification head
   - **Fine-tuned** - Pretrained backbone, fine-tune all layers

3. **Dataset**: ImageNette (10 easy ImageNet classes, ~9.5k train / ~3.9k val, 320px)

4. **Compute Budget**: 200 Google Colab Pro units (~17-25 units estimated usage)

5. **Timeline**: 4 weeks, ~1-2 days of actual implementation work

### Project Schema Highlights

- **Architecture**: ViT-Small/16 (~22M params)
- **Pretrained weights**: `google/vit-base-patch16-224` from HuggingFace
- **Framework**: PyTorch + HuggingFace transformers + datasets
- **Image size**: 224x224

#### Training Configuration

| Hyperparameter | From Scratch | Linear Probe | Fine-tuned |
|---------------|-------------|--------------|------------|
| Epochs        | 50-100      | 20-30        | 10-20      |
| Learning Rate | 1e-3        | 1e-2         | 2e-5       |
| Optimizer     | AdamW       | AdamW        | AdamW      |
| Batch Size    | 64          | 128          | 32         |
| Frozen Layers | None        | All except head | None    |

#### Evaluation Metrics
- Top-1 and Top-5 Accuracy
- Training time (wall clock + GPU hours)
- Convergence curves (loss and accuracy per epoch)
- Confusion matrix per variant
- Attention map visualization

#### Notebook Structure
1. Setup & Imports
2. Dataset Loading & Exploration
3. Data Preprocessing & Augmentation
4. Model Definition (ViT variants)
5. Training Loop (shared, configurable)
6. Experiment 1: ViT from Scratch
7. Experiment 2: ViT Linear Probe
8. Experiment 3: ViT Fine-tuned
9. Results Comparison (tables + plots)
10. Attention Map Visualization
11. Discussion & Conclusions

### Risks Identified
- ViT from scratch will likely underperform badly (expected - demonstrates data hunger)
- Colab disconnects: mitigate with checkpoints every 5 epochs to Google Drive
- If ImageNette is too easy: switch to ImageWoof (10 hard dog breed classes)

### Expected Outcome
Fine-tuned >> Linear Probe >> From Scratch, clearly demonstrating that ViT benefits enormously from pretraining and is data-hungry by nature.
