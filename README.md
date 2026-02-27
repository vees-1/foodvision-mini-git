# 🍕 FoodVision Mini

FoodVision Mini is an end-to-end computer vision project built with PyTorch, where an EfficientNet-B2 feature extractor is trained to classify food images into pizza, steak, or sushi.

The project demonstrates the full machine learning workflow — from modular training code and experimentation to deployment as a live web application.

---

## Live Demo

👉 Try the model live on Hugging Face Spaces:
https://huggingface.co/spaces/veees/FoodVision-mini

Upload an image or select an example to see real-time predictions.

---

## Model Performance

- Test Accuracy: 96.25%
- Dataset: FoodVision Mini (pizza, steak, sushi)
- Approach: EfficientNet-B2 feature extractor with frozen backbone
  
---

## Project Highlights

- Achieved 96.25% test accuracy on the FoodVision Mini dataset  
- Trained an EfficientNet-B2 feature extractor using transfer learning  
- Built a modular PyTorch training pipeline with clean separation of concerns  
- Followed ML engineering best practices for reproducibility and structure  
- Deployed an interactive Gradio web application on Hugging Face Spaces  
- Clearly separated training code (GitHub) from deployment code (Hugging Face)
---

## Repository Structure
```python
FoodVision-Mini/
│
├── src/            # Modular PyTorch training code
│   ├── data_setup.py
│   ├── engine.py
│   ├── helper_functions.py
│   ├── predictions.py
│   └── utils.py
│
├── notebooks/                # Experiments & exploration
│   └── FoodVision_Mini.ipynb
│
├── requirements.txt          # Training dependencies
├── .gitignore
├── LICENSE
└── README.md
```

**Note:**  
Deployment-specific files (`app.py`, model weights, Gradio UI) live in a separate Hugging Face Space repository, not in this GitHub repo.

---

## Dataset

The model is trained on a mini version of the FoodVision dataset, containing images from three food categories:

-  Pizza  
-  Steak  
-  Sushi  

The dataset is split into training and test sets and loaded using `torchvision.datasets.ImageFolder`.

---

## Model Architecture

- Base model: EfficientNet-B2 (pretrained on ImageNet)  
- Approach: Transfer learning (feature extractor)  
- Classifier: Custom fully connected classification head  
- Loss function: Cross-Entropy Loss  
- Optimizer: Adam  

Only the classifier head is trained while the pretrained backbone remains frozen.

---

## Training & Experiments

All training logic is implemented using a modular design:

- `data_setup.py` → dataset loading & transforms  
- `engine.py` → training and evaluation loops  
- `utils.py` → utility functions (saving models, helpers)  
- `predictions.py` → inference utilities  

This structure makes the code:
- reusable
- easy to debug
- simple to extend to new datasets or architectures

---

## Reproducibility

To install training dependencies:

```bash
pip install -r requirements.txt
```

Run experiments using the notebook:
```bash
notebooks/FoodVision_Mini.ipynb
```

## 🌐 Deployment

The trained model is deployed using:

- Gradio for the web interface  
- Hugging Face Spaces for hosting  
- Git LFS for managing large model files  

👉 Deployment repository:
https://huggingface.co/spaces/veees/FoodVision-mini

---

## Skills Demonstrated

- PyTorch & TorchVision  
- Transfer learning  
- Modular ML code design  
- Model evaluation & inference  
- Gradio application development  
- Hugging Face Spaces deployment  
- Git, Git LFS, and project structuring  

---

## 📄 License

This project is licensed under the MIT License.  
See the [`LICENSE`](LICENSE) file for details.

---

## 🙌 Acknowledgements

- PyTorch & TorchVision  
- Gradio  
- Hugging Face Spaces  
- EfficientNet (RWightman implementation)
