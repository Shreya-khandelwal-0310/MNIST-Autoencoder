# MNIST Image Denoising using Autoencoder

## Objective

This project builds a Deep Learning Autoencoder capable of removing random noise from handwritten MNIST digit images.

---

## Dataset

MNIST Handwritten Digits Dataset

- Training Images : 60,000
- Test Images : 10,000
- Image Size : 28 × 28

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

## Project Structure

```
MNIST-AUTOENCODER
│
├── dataset
├── assets
├── model.py
├── train.py
├── predict.py
├── autoencoder.ipynb
├── autoencoder.pth
├── requirements.txt
└── README.md
```

---

## Model Architecture

Encoder

28×28 → 128 → 64 → 32

Decoder

32 → 64 → 128 → 28×28

---

## Loss Function

Mean Squared Error (MSE)

---

## Optimizer

Adam Optimizer

Learning Rate = 0.001

---

## Training

```
python train.py
```

---

## Prediction

```
python predict.py
```

---

## Output

The model successfully reconstructs noisy MNIST images and removes most of the added Gaussian noise.

---

## Author

Shreya Khandelwal