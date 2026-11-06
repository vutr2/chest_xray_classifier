---
library_name: pytorch
pipeline_tag: image-classification
tags:
  - medical-imaging
  - chest-x-ray
  - densenet
  - grad-cam
metrics:
  - roc_auc
---

# Chest X-ray Nodule Classifier (DenseNet-121)

A DenseNet-121 fine-tuned to flag chest X-rays showing a **lung nodule or mass**, as opposed to a normal X-ray ("No Finding"). It was trained on NIH ChestX-ray14.

**Live demo:** [chest-xray-nodule-demo](https://huggingface.co/spaces/trungvu0307/chest-xray-nodule-demo) · **Code, training notebooks and full evaluation:** [github.com/vutr2/chest-xray-nodule-classifier](https://github.com/vutr2/chest-xray-nodule-classifier)

> Research and educational use only. Not a medical device and not for clinical use.

## Test-set performance

The test set has 6,966 X-rays from 2,917 patients who were never seen during training or model selection. The 95% confidence intervals come from a patient-level bootstrap.

| Metric | Value | 95% CI |
|---|---|---|
| AUC-ROC | 0.812 | 0.794 – 0.827 |
| Sensitivity | 90.3% | 88.1 – 91.9% |
| Specificity | 45.2% | 43.2 – 47.2% |

Sensitivity and specificity are measured at the stored threshold of 0.120, which was chosen on validation for 90% sensitivity.

## How to use

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models

ckpt = torch.load(hf_hub_download("trungvu0307/chest-xray-nodule-densenet121", "model.pt"), map_location="cpu")
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

- **Input:** the X-ray converted to a 3-channel RGB image, resized to 224×224, with ImageNet mean/std normalisation.
- **Output:** one logit, where `sigmoid(logit)` is the nodule/mass probability.
- **Checkpoint contents:** `threshold`, `img_size`, `epoch`, `val_auc` and the training settings (`lr`, `weight_decay`).

## Training

- **Data:** 44,828 X-rays from NIH ChestX-ray14. Positives are every X-ray labelled Nodule or Mass. Negatives are "No Finding" X-rays, sampled at 3 per positive.
- **Split:** by patient into train, validation and test sets.
- **Model:** starts from ImageNet weights.
- **Training setup:**
  - Loss: BCE with `pos_weight` ≈ 3.
  - Optimiser: AdamW (lr 3e-5, weight decay 1e-2) with a cosine schedule.
  - Augmentation: crops, rotations, shifts, and brightness and contrast changes.
  - Early stopping on validation AUC, with the best result at epoch 6.

## Limitations

- **Noisy labels:** the labels were text-mined from radiology reports.
- **Normal X-rays only as negatives:** X-rays with other diseases and no nodule were never seen, so the model is likely to flag many of them.
- **Weaker on some cases:** sensitivity is lower for isolated nodules (86.8%), and AUC is lower on AP/portable X-rays (0.773).
- **One source:** single hospital, no external validation.

## Citation

Data provided by the NIH Clinical Center.

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* CVPR 2017.
