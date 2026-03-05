#  *`FoodVision Mini`*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Huggingface-Live%20Demo-yellow)](https://huggingface.co/spaces/veees/FoodVision-mini)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.25%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Domain: Computer Vision · Transfer Learning · Model Deployment*    
*Dataset: Food101 subset (Pizza · Steak · Sushi)*    
*Framework: PyTorch · TorchVision · Gradio*

👉 [*Try it live on Hugging Face Spaces*](https://huggingface.co/spaces/veees/FoodVision-mini)

---
## Overview

*FoodVision Mini is a 3-class food image classifier achieving **96.25% test accuracy** via transfer learning on a small dataset. It fine-tunes an **EfficientNet-B2** backbone pretrained on ImageNet — keeping the feature extractor frozen and training only a lightweight classification head — then deploys to a real-time Gradio app on Hugging Face Spaces.*

---

## Model Architecture

```
Input Image (224×224×3)
        │
EfficientNet-B2 Backbone  ← Frozen (ImageNet pretrained)
        │
Dropout → Linear(1408 → 3)  ← Trained
        │
Softmax → [Pizza, Steak, Sushi]
```

---

## Dataset

*A mini subset of [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) with three categories: **Pizza**, **Steak**, and **Sushi**. Images are loaded via `torchvision.datasets.ImageFolder` with augmentation transforms (random flips, color jitter) applied during training and center-crop normalization at test time.*

---


## Getting Started

```bash
git clone https://github.com/yourusername/FoodVision-Mini.git
cd FoodVision-Mini
pip install -r requirements.txt
jupyter notebook notebooks/FoodVision_Mini.ipynb
```

---

## Deployment

*The trained model is served via a **Gradio** web app on Hugging Face Spaces. Model weights are tracked with Git LFS and the Space auto-rebuilds on every push.*

👉 [*Deployment repo*](https://huggingface.co/spaces/veees/FoodVision-mini)

