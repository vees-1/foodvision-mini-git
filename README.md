# FoodVision Mini 🍕🥩🍣

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-yellow)](https://huggingface.co/spaces/veees/FoodVision-mini)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.25%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end computer vision project built with PyTorch.

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/veees/FoodVision-mini)

Upload any food image or pick an example to get real-time predictions.

---

## Project Highlights

- Achieved 96.25% test accuracy using transfer learning on a small 3-class dataset
- Fine-tuned EfficientNet-B2 pretrained on ImageNet — only the classifier head is trained
- Built a modular PyTorch training pipeline with clean separation of concerns across data, engine, utils, and inference
- Deployed an interactive Gradio app on Hugging Face Spaces with real-time inference
- Cleanly separated training code (this repo) from deployment code (Hugging Face Space)

---

## Model Architecture

The model uses EfficientNet-B2 pretrained on ImageNet as a frozen feature extractor, with a custom fully connected classification head trained on top.

```
Input Image
     │
EfficientNet-B2 Backbone (frozen)
     │
Custom Classifier Head (trained)
     │
Softmax → [Pizza, Steak, Sushi]
```

- Loss: Cross-Entropy Loss
- Optimizer: Adam
- Backbone: Frozen (feature extractor mode)
- Head: Trainable linear layers

---

## Dataset

The model is trained on a mini subset of the Food101 dataset containing three categories:

- 🍕 Pizza
- 🥩 Steak
- 🍣 Sushi

Images are loaded using `torchvision.datasets.ImageFolder` with standard train/test splits and augmentation transforms applied during training.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/FoodVision-Mini.git
cd FoodVision-Mini

# Install dependencies
pip install -r requirements.txt
```

Run the full experiment in the notebook:

```bash
jupyter notebook notebooks/FoodVision_Mini.ipynb
```

---

## Deployment

The trained model is served via a Gradio web app hosted on Hugging Face Spaces.

- Model weights are tracked with Git LFS
- The Space auto-rebuilds on every push to the HF repo
- Inference runs in real-time directly in the browser

👉 [Deployment repo](https://huggingface.co/spaces/veees/FoodVision-mini)

---

## Skills Demonstrated

PyTorch & TorchVision · Transfer Learning · Modular ML Pipeline Design · Model Evaluation · Gradio · Hugging Face Spaces · Git LFS · Reproducible Project Structure


## License

MIT License — see [LICENSE](LICENSE) for details.
