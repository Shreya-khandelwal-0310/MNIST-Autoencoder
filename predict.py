import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import Autoencoder

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

# Load test dataset
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./dataset",
    train=False,
    download=True,
    transform=transform
)

# Select one image
image, _ = test_dataset[0]
image = image.unsqueeze(0).to(device)

# Add noise
noise = torch.randn_like(image) * 0.5
noisy_image = torch.clamp(image + noise, 0., 1.)

# Predict
with torch.no_grad():
    output = model(noisy_image)

# Convert to numpy
original = image.squeeze().cpu().numpy()
noisy = noisy_image.squeeze().cpu().numpy()
denoised = output.squeeze().cpu().numpy()

# Display images
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(original, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(denoised, cmap="gray")
plt.title("Denoised")
plt.axis("off")

plt.savefig("assets/result.png")
print("Prediction image saved in assets/result.png")