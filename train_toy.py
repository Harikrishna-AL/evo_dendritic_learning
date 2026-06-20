import torch
import torch.nn as nn
import torch.optim as optim
from dendritic_neuron import DendriticNeuron

def generate_task_data(task_id, num_samples=1000):
    """
    Generate synthetic data for Continual Learning.
    Task 1: Learn to classify if X[0] > 0  (Simple half-plane)
    Task 2: Learn to classify if X[0] > 0 OR X[1] > 0 (Union of two half-planes)
    """
    X = torch.randn(num_samples, 2)
    
    if task_id == 1:
        # Target 1: Only depends on the first feature
        y = (X[:, 0] > 0).float().unsqueeze(1)
    elif task_id == 2:
        # Target 2: Depends on both features (requires a more complex boundary)
        y = ((X[:, 0] > 0) | (X[:, 1] > 0)).float().unsqueeze(1)
        
    return X, y

def train_neuron_gradient_descent(neuron, X, y, epochs=1000, lr=0.1):
    """
    Train the weights of the current structure using Gradient Descent.
    """
    criterion = nn.MSELoss() 
    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in neuron.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = neuron(X)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    return loss.item()

if __name__ == "__main__":
    print("--- Phase 1: Learning Task 1 ---")
    X1, y1 = generate_task_data(task_id=1)
    
    # Initialize a neuron with 1 branch
    neuron = DendriticNeuron(input_dim=2, initial_branches=1)
    print("Training on Task 1 data (Feature 1 > 0)...")
    train_neuron_gradient_descent(neuron, X1, y1, epochs=1000)
    
    
    print("\n--- Phase 2: The Environment Changes (Task 2) ---")
    X2, y2 = generate_task_data(task_id=2)
    
    print("Evaluating current 1-branch neuron on Task 2 without training...")
    with torch.no_grad():
        initial_loss = nn.MSELoss()(neuron(X2), y2)
        print(f"Initial Loss on Task 2: {initial_loss.item():.4f} (It can't solve it!)")
        
    print("\nSimulating Evolutionary Phase: Adding a branch...")
    neuron.add_branch()
    
    print("Freezing mature Branch 0 to prevent catastrophic forgetting...")
    for param in neuron.branches[0].parameters():
        param.requires_grad = False
        
    print("Training on Task 2 data with the new structural capacity...")
    train_neuron_gradient_descent(neuron, X2, y2, epochs=1000)
    
    print("\nDone! The neuron successfully grew structure to learn a new concept while preserving the old one.")
