# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:19:30 2025

@author: taske
"""

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