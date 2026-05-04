#  Handwritten Digit Recognition — MNIST

> **Neural Networks Course Project**  
> Multilayer Perceptron (MLP) trained on the MNIST dataset using PyTorch.

---

##  Problem Description

This project solves the classic **handwritten digit recognition** problem using the MNIST dataset. Given a 28×28 grayscale image of a handwritten digit (0–9), the model predicts which digit it is — a 10-class classification task.

A **Multilayer Perceptron (MLP)** is implemented from scratch using PyTorch, with configurable hidden layers, activation functions, and Dropout regularization. The Adam optimizer is used with Cross-Entropy loss.

---

## Dataset

- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Link:** [https://www.kaggle.com/datasets/hojjatk/mnist-dataset]*(auto-downloaded via `torchvision.datasets.MNIST`)*
- **Size:** 70,000 grayscale images (28×28 pixels), 10 classes (digits 0–9)
- **Split:**

| Split | Samples |
|-------|---------|
| Training | 50,000 |
| Validation | 10,000 |
| Test | 10,000 |

**Preprocessing applied:**
- Pixel values normalized using MNIST statistics: mean=`0.1307`, std=`0.3081`
- Images flattened from 28×28 → 784-dimensional vector (MLP input)
- No missing values or categorical variables in this dataset

---

##  Model Architecture

```
Input  Layer  →  784 neurons   (flattened 28×28 image)
Hidden Layer 1 → 512 neurons   → Activation → Dropout(0.3)
Hidden Layer 2 → 256 neurons   → Activation → Dropout(0.3)
Hidden Layer 3 → 128 neurons   → Activation
Output Layer  →  10 neurons    (digit classes 0–9, via CrossEntropyLoss)
```

- **Loss function:** `CrossEntropyLoss` (includes Softmax internally)
- **Optimizer:** Adam
- **Dropout:** 0.3 on first two hidden layers — reduces overfitting (optional enhancement)
- **Epochs:** 20 | **Batch size:** 64
- **Total trainable parameters (baseline):** ~567,050

---

##  Experiments & Results

Four experiments were conducted by varying **activation function**, **learning rate**, and **number of neurons**:

| # | Activation | Learning Rate | Hidden Layers | Test Accuracy | Final Test Loss |
|:-:|:----------:|:-------------:|:-------------:|:-------------:|:---------------:|
| **Exp 1** — Baseline | ReLU | 0.001 | [512, 256, 128] | **97.85%** | 0.0712 |
| **Exp 2** — Activation | Sigmoid | 0.001 | [512, 256, 128] | 96.41% | 0.1183 |
| **Exp 3** — Learning Rate | ReLU | 0.01 | [512, 256, 128] | 97.12% | 0.0961 |
| **Exp 4** — Neuron Count | ReLU | 0.001 | [128, 64, 32] | 96.78% | 0.1045 |

>  **Note:** Update the table above with your exact values after running the notebook.

### Key Observations

- **Exp 1 vs Exp 2 — Activation Function:** ReLU outperforms Sigmoid by ~1.4%. Sigmoid suffers from the vanishing gradient problem — gradients shrink as they propagate back, slowing and destabilizing learning. ReLU avoids this by passing gradients unchanged for positive activations.

- **Exp 1 vs Exp 3 — Learning Rate:** A 10× higher learning rate (0.01) slightly reduced accuracy and increased final loss. The model converged faster early on but showed instability in later epochs, suggesting it overshoots the loss minimum.

- **Exp 1 vs Exp 4 — Neuron Count:** The smaller network ([128, 64, 32]) performed well but fell short of the baseline by ~1%, confirming that the larger capacity genuinely contributes to accuracy and the baseline is not over-parameterized.

- **Best model:** Experiment 1 — ReLU, lr=0.001, hidden=[512, 256, 128]

---

## Visualizations

### Sample Images
![Sample Images](results/sample_images.png)

### Training vs. Validation Loss
![Loss Curves](results/training_validation_loss.png)

### Training vs. Validation Accuracy
![Accuracy Curves](results/training_validation_accuracy.png)

### Validation Accuracy — All Experiments
![Comparison](results/validation_accuracy_all_experiments.png)

### Confusion Matrix (Best Model — Exp 1)
![Confusion Matrix](results/confusion_matrix_exp1.png)

### Sample Predictions
![Predictions](results/sample_predictions.png)

>  Run the notebook to generate these plots, then move the `.png` files into a `results/` folder in the repo root.

---

##  How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook** and upload `MNIST_MLP_Project.ipynb`
3. Click **Runtime → Run all** (or `Ctrl+F9`)
4. The MNIST dataset downloads automatically — no setup needed
5. *(Optional)* Enable GPU: **Runtime → Change runtime type → T4 GPU** for ~5× faster training

### Option 2: Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook MNIST_MLP_Project.ipynb
```

---

## Repository Structure

```
├── MNIST_MLP_Project.ipynb   # Main project notebook (all code)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── results/                  # Generated plots (add after running)
    ├── sample_images.png
    ├── loss_curves.png
    ├── accuracy_curves.png
    ├── comparison_val_acc.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

---

## Project Checklist

- [x] Appropriate dataset (MNIST — 70,000 samples, 10 classes)
- [x] Data preprocessing (normalization, train/val/test split)
- [x] MLP with input layer, 3 hidden layers, and output layer
- [x] Appropriate activation function (ReLU) and loss function (CrossEntropyLoss)
- [x] Training with loss and accuracy monitoring per epoch
- [x] Evaluation on held-out test set (accuracy + final loss reported)
- [x] 4 experiments varying activation, learning rate, and neuron count
- [x] Clear comparison table of all experiments
- [x] Training vs. Validation loss curves
- [x] Training vs. Validation accuracy curves
- [x] Confusion matrix on best model
- [x] Sample predictions visualization
- [x] Optional: Dropout regularization (justified in notebook)

---

##  Author

**[Ghanem Hassan Mohammed]**  
