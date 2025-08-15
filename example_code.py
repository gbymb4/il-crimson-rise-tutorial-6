# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:18:51 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

print("CNN Fundamentals: Understanding Convolutions with PyTorch")
print("=" * 55)

# Load a single MNIST sample for demonstration
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST('./data', train=False, download=True, transform=transform)
sample_image, label = mnist[0]  # Shape: (1, 28, 28)

print(f"Original MNIST digit: {label}")
print(f"Image shape: {sample_image.shape}")

# Create a batch for PyTorch (batch_size=1)
image_batch = sample_image.unsqueeze(0)  # Shape: (1, 1, 28, 28)
print(f"Batch shape: {image_batch.shape}")

# PART 1: Single Convolution Layer
print("\n" + "="*40)
print("PART 1: Single Convolution Layer")

# Create a conv layer with 3 filters
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
print(f"Conv layer: {conv_layer}")
print(f"Filter weights shape: {conv_layer.weight.shape}")

# Apply convolution
conv_output = conv_layer(image_batch)
print(f"Convolution output shape: {conv_output.shape}")

# Visualize the 3 feature maps
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

axes[0].imshow(sample_image.squeeze(), cmap='gray')
axes[0].set_title(f'Original ({label})')
axes[0].axis('off')

for i in range(3):
    axes[i+1].imshow(conv_output[0, i].detach(), cmap='gray')
    axes[i+1].set_title(f'Filter {i+1}')
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()

# PART 2: Convolution + ReLU + Pooling
print("\n" + "="*40) 
print("PART 2: Conv + ReLU + Pooling")

# Add ReLU activation
relu_output = F.relu(conv_output)
print(f"After ReLU: {relu_output.shape}")

# Add MaxPooling
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
pooled_output = pool_layer(relu_output)
print(f"After MaxPool2d: {pooled_output.shape}")

# Show the effect of pooling
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for i in range(3):
    # Before pooling
    axes[0, i].imshow(relu_output[0, i].detach(), cmap='gray')
    axes[0, i].set_title(f'Before Pool: {relu_output.shape[2:]}')
    axes[0, i].axis('off')
    
    # After pooling  
    axes[1, i].imshow(pooled_output[0, i].detach(), cmap='gray')
    axes[1, i].set_title(f'After Pool: {pooled_output.shape[2:]}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# PART 3: Multiple Conv Layers
print("\n" + "="*40)
print("PART 3: Stacking Conv Layers")

# Build a simple CNN block
conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)    # 1->16 filters
conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 16->32 filters
pool = nn.MaxPool2d(2, 2)

print("CNN Architecture:")
print(f"Input: {image_batch.shape}")

# Forward pass through layers
x = image_batch
print(f"Input: {x.shape}")

x = F.relu(conv1(x))
print(f"After conv1 + ReLU: {x.shape}")

x = pool(x)
print(f"After pool1: {x.shape}")

x = F.relu(conv2(x))
print(f"After conv2 + ReLU: {x.shape}")

x = pool(x)
print(f"After pool2: {x.shape}")

# Show how to flatten for FC layers
x_flattened = x.view(x.size(0), -1)
print(f"Flattened for FC layers: {x_flattened.shape}")

# PART 4: Complete Minimal CNN
print("\n" + "="*40)
print("PART 4: Complete Minimal CNN")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 32 filters, 7x7 spatial size
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))    # 14x14 -> 7x7
        x = x.view(-1, 32 * 7 * 7)              # Flatten
        x = self.fc(x)
        return x

# Create and test the model
model = SimpleCNN()
print(f"Model:\n{model}")

# Test forward pass
output = model(image_batch)
print(f"\nModel output shape: {output.shape}")
print(f"Raw output (logits): {output[0][:5]}...")  # Show first 5 values

# Convert to probabilities
probs = F.softmax(output, dim=1)
predicted_class = torch.argmax(probs, dim=1)
print(f"Predicted class: {predicted_class.item()}")
print(f"Actual class: {label}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

print("\nKey CNN Concepts:")
print("1. Convolution preserves spatial structure")  
print("2. Pooling reduces dimensions and adds translation invariance")
print("3. Multiple filters learn different features")
print("4. Stacking layers learns increasingly complex patterns")
print("5. Final FC layer maps features to classes")