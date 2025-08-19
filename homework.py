# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 21:29:03 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%% ============================================================================
# PART 1: DATA LOADING (REQUIRED)
# ============================================================================

# TODO 1.1: Define transforms for MNIST
# Hint: At minimum use transforms.ToTensor()
transform = None

# TODO 1.2: Load MNIST datasets 
train_dataset = None  # TODO: torchvision.datasets.MNIST()
test_dataset = None   # TODO: torchvision.datasets.MNIST()

# TODO 1.3: Create DataLoaders
# Choose your own batch_size
train_loader = None
test_loader = None

print(f"Train samples: {len(train_dataset) if train_dataset else 'TODO'}")
print(f"Test samples: {len(test_dataset) if test_dataset else 'TODO'}")

#%% ============================================================================
# PART 2: CNN IMPLEMENTATION (REQUIRED)
# ============================================================================

class CNN(nn.Module):
    """
    Design your own CNN architecture for MNIST.
    
    Requirements:
    - Must use at least 2 convolutional layers
    - Must use at least 1 pooling layer  
    - Must end with fully connected layer(s) for classification
    - Must output 10 classes for MNIST digits
    
    Architecture choices are entirely up to you:
    - Number of filters per layer
    - Kernel sizes
    - Padding strategies
    - Activation functions
    - Dropout rates
    - Number of FC layers
    """
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # TODO 2.1: Design your convolutional layers
        # Examples: nn.Conv2d(), nn.MaxPool2d(), nn.BatchNorm2d()
        
        # TODO 2.2: Design your fully connected layers
        # Remember to calculate the correct input size after conv operations
        
        # TODO 2.3: Add any regularization (dropout, batch norm, etc.)
        
    def forward(self, x):
        # TODO 2.4: Implement forward pass
        # Don't forget to flatten before FC layers!
        
        return x

# TODO 2.5: Create model instance
model = None  # TODO: Create CNN() and move to device

#%% ============================================================================
# PART 3: TRAINING SETUP (REQUIRED)
# ============================================================================

# TODO 3.1: Define loss function
criterion = None

# TODO 3.2: Define optimizer  
# Choose learning rate and optimizer type
optimizer = None

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, targets in train_loader:
        # TODO 3.3: Implement training step
        # - Move data to device
        # - Zero gradients  
        # - Forward pass
        # - Compute loss
        # - Backward pass
        # - Update weights
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    return total_loss / len(train_loader), 100.0 * correct / total

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model, return average loss and accuracy"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            # TODO 3.4: Implement evaluation step
            # - Move data to device
            # - Forward pass
            # - Compute loss
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return total_loss / len(test_loader), 100.0 * correct / total

#%% ============================================================================
# PART 4: TRAINING LOOP (REQUIRED)
# ============================================================================

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=10, device='cpu'):
    """Complete training loop with metrics tracking"""
    
    # TODO 4.1: Initialize metrics storage
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # TODO 4.2: Train and evaluate
        train_loss, train_acc = None, None  # TODO: Call train_epoch()
        test_loss, test_acc = None, None    # TODO: Call evaluate_model()
        
        # TODO 4.3: Store metrics
        # history['train_loss'].append(train_loss)
        # etc.
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%, Time {epoch_time:.1f}s")
    
    return history

# TODO 4.4: Train your model
# history = train_model(model, train_loader, test_loader, criterion, optimizer, 
#                      num_epochs=10, device=device)

#%% ============================================================================
# PART 5: VISUALIZATION (REQUIRED)
# ============================================================================

def plot_training_history(history):
    """Plot training curves"""
    
    # TODO 5.1: Create training curve plots
    # Plot both loss and accuracy over epochs
    # Include both training and test curves
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # TODO: Create subplots and plot curves
    # TODO: Add labels, legends, titles
    
    plt.tight_layout()
    plt.show()

# TODO 5.2: Generate required plots
# plot_training_history(history)

#%% ============================================================================
# ANALYSIS QUESTIONS (REQUIRED)
# ============================================================================

"""
TODO: Answer these questions after completing your implementation:

1. Final test accuracy: ____%

2. How many parameters does your model have? 
   Answer:

3. Describe your architectural choices and why you made them:
   Answer:

4. What challenges did you encounter during implementation?
   Answer:
"""

#%% ============================================================================
# OPTIONAL CHALLENGES (Extra Credit)
# ============================================================================

"""
Choose any of these optional challenges for extra learning (these will require
independent research):

ADVANCED ARCHITECTURES:
- Implement residual connections (ResNet-style skip connections)
- Add attention mechanisms (spatial or channel attention)
- Try batch normalization layers

FILTER VISUALIZATION:
- Plot learned convolutional filters 
- Visualize feature maps at different layers
- Create activation heatmaps

PERFORMANCE OPTIMIZATION:
- Experiment with different optimizers and learning rates
- Add learning rate scheduling
- Implement data augmentation
- Try different regularization techniques

ANALYSIS:
- Compare different CNN architectures
- Analyze per-class performance
- Study training dynamics and convergence

Choose one or as many challenges as you like and implement them. Document your 
approach and results.
"""

print("\nOptional challenges available for extra credit!")
print("See comments above for ideas.")

#%% ============================================================================
# SUBMISSION CHECKLIST
# ============================================================================

print("\nSUBMISSION CHECKLIST:")
print("□ Working CNN implementation")
print("□ Successful training with >95% test accuracy")
print("□ Training curve plots")
print("□ Completed analysis questions")
print("□ (Optional) Extra credit challenge")

print("\nDeadline: [Insert due date]")
print("Submit: .py file + plots + written answers")