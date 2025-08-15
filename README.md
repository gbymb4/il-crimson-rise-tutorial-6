# Machine Learning Session 6: Convolutional Neural Networks
## From MLPs to CNNs - Understanding Image Feature Learning

### Session Overview
**Duration**: 1 hour  
**Prerequisites**: Completed Session 5 (PyTorch Neural Networks/MLPs)  
**Goal**: Understand convolutions and build CNNs for image classification  
**Focus**: Convolution operations, spatial feature learning, and CNN architecture

### Session Timeline
| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:05 | 1. Touching Base & Session Overview    |
| 0:05 - 0:30 | 2. Understanding CNNs - PyTorch Demo |
| 0:30 - 0:55 | 3. Solo Exercise: MLP to CNN Conversion |
| 0:55 - 1:00 | 4. Wrap-up & Key Insights |

---

## 1. Touching Base & Session Overview (5 minutes)

### Quick Check-in
- Review Session 5's MLP implementation for MNIST
- Discuss limitations of fully connected layers for image data
- Preview today's focus on spatial feature learning

### Today's Learning Objectives
By the end of this session, you will be able to:
- Understand what convolutions are and how they work on images
- Explain why CNNs are better than MLPs for image tasks
- Implement CNN layers using PyTorch (`nn.Conv2d`, `nn.MaxPool2d`)
- Convert an existing MLP architecture to a CNN
- Compare CNN vs MLP performance on image classification

### The Core Question
**Why do we need CNNs when MLPs work for MNIST?**
- MLPs treat images as flat vectors (lose spatial structure)
- CNNs preserve spatial relationships and learn local patterns
- Translation invariance: same feature detected anywhere in image
- Parameter efficiency: shared filters vs unique weights for each pixel

---

## 2. Understanding CNNs - PyTorch Demo (25 minutes)

### What is a Convolution? (5 minutes)

A convolution is a mathematical operation that:
1. **Slides a filter** (small matrix) across an image
2. **Computes dot products** between filter and image patches
3. **Produces a feature map** showing where patterns are detected

**Key Insight**: Instead of learning separate weights for each pixel position, we learn a few filters that detect important patterns anywhere in the image.

### Live CNN Demo with PyTorch (20 minutes)

*This interactive script demonstrates CNN concepts using PyTorch layers directly*

```python
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
```

### Key Points During Demo

**1. Shape Transformations**
- Input: `(batch, channels, height, width)` 
- Conv2d preserves spatial dimensions (with padding)
- MaxPool2d reduces spatial dimensions by factor of kernel_size
- Flatten before FC layers: `x.view(-1, flattened_size)`

**2. Parameter Efficiency**
- Conv layer with 32 filters of size 3×3: only 32 × 3 × 3 = 288 weights per input channel
- Equivalent MLP would need 784 × hidden_size weights
- Same filters applied everywhere = translation invariance

**3. Feature Learning Hierarchy**
- Early layers: simple edges and textures
- Later layers: more complex patterns
- FC layer: combines spatial features for classification

---

## 3. Solo Exercise: MLP to CNN Conversion (25 minutes)

### Exercise Overview
Convert a working MLP implementation for MNIST into a CNN. This hands-on approach helps you understand architectural differences and implementation details.

### Starting Point: Working MLP (Provided)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
print("MLP to CNN Conversion Exercise")
print("=" * 35)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading (identical for both MLP and CNN)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Working MLP (provided as baseline)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training function (works for both MLP and CNN)
def train_model(model, train_loader, test_loader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_acc = 100 * correct / total
        
        # Test accuracy
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()
        
        test_acc = 100 * correct_test / total_test
        print(f'Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')
    
    return test_acc

# Train MLP baseline
print("\nTraining MLP baseline...")
mlp_model = MLP().to(device)
mlp_accuracy = train_model(mlp_model, train_loader, test_loader)
mlp_params = sum(p.numel() for p in mlp_model.parameters())
print(f"MLP: {mlp_accuracy:.1f}% accuracy, {mlp_params:,} parameters")
```

### Your Task: Implement CNN

```python
#%% TODO: Implement CNN to beat the MLP
print("\n" + "="*40)
print("YOUR TASK: Implement CNN")
print("="*40)

class CNN(nn.Module):
    """Convolutional Neural Network for MNIST"""
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # TODO 1: Define convolutional layers
        # Hint: nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # MNIST: 1 input channel (grayscale), try 16->32 output channels
        
        self.conv1 = None  # TODO: 1 -> 16 filters, 3x3 kernel
        self.conv2 = None  # TODO: 16 -> 32 filters, 3x3 kernel
        
        # TODO 2: Define pooling layer
        # Hint: nn.MaxPool2d(kernel_size=2, stride=2) halves spatial dimensions
        
        self.pool = None   # TODO: 2x2 max pooling
        
        # TODO 3: Define fully connected layers
        # Challenge: Calculate flattened size after conv+pool operations
        # MNIST 28x28 -> conv -> pool -> conv -> pool -> ??x??
        # Hint: 28 -> 28 -> 14 -> 14 -> 7, with 32 channels = 32*7*7
        
        self.fc1 = None    # TODO: flattened_size -> 128 hidden units
        self.fc2 = None    # TODO: 128 -> 10 classes
        
        # TODO 4: Define dropout (optional)
        self.dropout = None
        
    def forward(self, x):
        # TODO 5: Implement forward pass
        # Pattern: conv -> relu -> pool -> conv -> relu -> pool -> flatten -> fc
        
        # TODO 5a: First conv block
        # Apply conv1, then ReLU activation, then pooling
        x = None
        
        # TODO 5b: Second conv block  
        # Apply conv2, then ReLU activation, then pooling
        x = None
        
        # TODO 5c: Flatten for fully connected layers
        # Hint: x.view(x.size(0), -1) keeps batch dimension, flattens rest
        x = None
        
        # TODO 5d: Fully connected layers
        # Apply fc1, ReLU, dropout (if used), then fc2
        x = None
        
        return x

# TODO 6: Helper function to calculate conv output size
def calculate_conv_size():
    """
    Helper to figure out flattened size after conv operations
    Run this to determine the size for your first FC layer
    """
    # TODO: Create a dummy CNN with your conv layers defined
    # Pass a dummy input through just the conv layers
    # Print the shape to determine flattened size
    
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch of 1, MNIST shape
    
    # TODO: Apply your conv layers and print shapes
    print("Dummy input shape:", dummy_input.shape)
    # x = your_conv_operations(dummy_input)
    # print("After conv operations:", x.shape)
    # flattened = x.view(1, -1) 
    # print("Flattened size:", flattened.shape[1])
    
    pass  # Remove when implemented

# TODO 7: Create and test CNN
print("TODO 7: Create and test your CNN")

# TODO 7a: Uncomment when ready to test
# calculate_conv_size()  # Use this to find your FC layer size

# TODO 7b: Create CNN instance  
cnn_model = None  # TODO: Create CNN() and move to device

# TODO 7c: Compare model sizes
# cnn_params = sum(p.numel() for p in cnn_model.parameters())
# print(f"\nModel comparison:")
# print(f"MLP: {mlp_params:,} parameters")  
# print(f"CNN: {cnn_params:,} parameters")

# TODO 8: Train CNN and compare results
print("TODO 8: Train your CNN")

# TODO 8a: Train CNN
# print("Training CNN...")
# cnn_accuracy = train_model(cnn_model, train_loader, test_loader)

# TODO 8b: Final comparison
# print(f"\nFinal Results:")
# print(f"MLP: {mlp_accuracy:.1f}% accuracy, {mlp_params:,} parameters")
# print(f"CNN: {cnn_accuracy:.1f}% accuracy, {cnn_params:,} parameters")
# print(f"Improvement: {cnn_accuracy - mlp_accuracy:.1f} percentage points")

#%% TODO 9: Analysis Questions (Answer after implementation)
"""
After implementing and training your CNN, answer these questions:

1. ACCURACY: Did your CNN outperform the MLP? By how much?

2. PARAMETERS: Does the CNN have more or fewer parameters than the MLP?
   Why might this be surprising?

3. ARCHITECTURE: What does each part of your CNN do?
   - Conv layers: 
   - Pooling: 
   - FC layers:

4. FEATURE LEARNING: What might your conv filters be learning?
   How is this different from the MLP approach?

5. GENERALIZATION: Why might CNNs generalize better to new images
   compared to MLPs?

6. CHALLENGES: What was the trickiest part of implementing the CNN?
   What would you do differently next time?
"""

print("\nHints if you're stuck:")
print("- Start with calculate_conv_size() to find FC layer input size")
print("- Common CNN pattern: 1->16->32 channels, 3x3 kernels, padding=1")  
print("- Don't forget ReLU activations after conv layers")
print("- MaxPool2d(2,2) halves spatial dimensions: 28->14->7")
print("- Final FC layer maps features to 10 classes (digits 0-9)")
```

### Expected Implementation Hints

**Shape Calculations**
- MNIST: 28×28 input
- After conv1 (padding=1): still 28×28, but 16 channels
- After pool1: 14×14×16
- After conv2 (padding=1): still 14×14, but 32 channels  
- After pool2: 7×7×32 = 1568 flattened size

**Common Mistakes to Watch For**
- Forgetting to add padding (causes dimension mismatches)
- Wrong flattened size calculation
- Missing ReLU activations
- Incorrect tensor reshaping

**Success Criteria**
- CNN should achieve >98% accuracy (vs ~97% for MLP)
- CNN should have fewer parameters than MLP
- Code should run without shape errors

---

## 4. Wrap-up & Key Insights (5 minutes)

### Key Takeaways
- **Spatial Structure Matters**: CNNs preserve spatial relationships that MLPs destroy
- **Parameter Efficiency**: Shared conv filters use fewer parameters than fully connected layers
- **Translation Invariance**: Same features detected regardless of position in image
- **Hierarchical Learning**: Early layers learn simple features, later layers combine them

### CNN vs MLP Comparison
- **MLP**: Treats image as flat vector, learns position-specific weights
- **CNN**: Preserves spatial structure, learns location-invariant features
- **Performance**: CNNs typically achieve higher accuracy with fewer parameters on image tasks

### Next Steps
- **Session 7**: Advanced CNN architectures (deeper networks, residual connections)
- **Future topics**: Transfer learning, data augmentation, other domains (text, audio)

### Questions for Reflection
- Why do CNNs work better for images than MLPs?
- How does the CNN feature hierarchy relate to human vision?
- When might you still prefer MLPs over CNNs?
- What other types of data might benefit from convolutional approaches?