import torch
import torch.nn as nn
import torch.optim as optim
import copy
from dendritic_neuron import DendriticNeuron

def generate_task_data(task_id, num_samples=1000):
    """
    Task 1: Learn to classify if X[0] > 0
    Task 2: Learn to classify if X[0] > 0 OR X[1] > 0
    """
    X = torch.randn(num_samples, 2)
    if task_id == 1:
        y = (X[:, 0] > 0).float().unsqueeze(1)
    elif task_id == 2:
        y = ((X[:, 0] > 0) | (X[:, 1] > 0)).float().unsqueeze(1)
    return X, y

def train_continuous_weights(neuron, X, y, epochs=1000, lr=0.1):
    """
    Continuous Phase (Gradient Descent):
    Train the weights of the current structure.
    """
    criterion = nn.MSELoss() 
    trainable_params = [p for p in neuron.parameters() if p.requires_grad]
    if not trainable_params:
        return 0.0
    
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

def evolutionary_mask_search(neuron, X, y, generations=50, pop_size=20, target_loss=0.01):
    """
    Discrete Phase (Evolutionary Algorithm):
    Search for a binary mask M over the inputs and branches that minimizes loss.
    Returns True if a successful mask was found, False otherwise.
    """
    criterion = nn.MSELoss()
    
    # Extract current masks
    masks = []
    for branch in neuron.branches:
        masks.append((branch.input_mask.clone(), branch.active_mask.clone()))
        
    best_loss = float('inf')
    best_masks = None
    
    print(f"Starting EA search over binary masks...")
    
    for gen in range(generations):
        # Generate a population of random binary masks
        for i in range(pop_size):
            test_neuron = copy.deepcopy(neuron)
            
            # Mutate masks randomly
            for branch in test_neuron.branches:
                branch.input_mask.data = torch.randint(0, 2, branch.input_mask.shape).float()
                branch.active_mask.data = torch.randint(0, 2, branch.active_mask.shape).float()
                
            with torch.no_grad():
                loss = criterion(test_neuron(X), y).item()
                
            if loss < best_loss:
                best_loss = loss
                best_masks = []
                for branch in test_neuron.branches:
                    best_masks.append((branch.input_mask.clone(), branch.active_mask.clone()))
                    
        if best_loss <= target_loss:
            print(f"EA found a solution at generation {gen+1}! Loss: {best_loss:.4f}")
            # Apply the best mask to the actual neuron
            for idx, branch in enumerate(neuron.branches):
                branch.input_mask.data, branch.active_mask.data = best_masks[idx]
            return True
            
    print(f"EA failed to find a perfect mask. Best loss: {best_loss:.4f}")
    return False

if __name__ == "__main__":
    print("--- Phase 1: Learning Task 1 ---")
    X1, y1 = generate_task_data(task_id=1)
    
    neuron = DendriticNeuron(input_dim=2, initial_branches=1)
    print("Training continuous weights on Task 1...")
    train_continuous_weights(neuron, X1, y1, epochs=1000)
    
    print("\nFreezing Task 1 continuous weights permanently...")
    for param in neuron.parameters():
        param.requires_grad = False
        
    print("\n--- Phase 2: The Environment Changes (Task 2) ---")
    X2, y2 = generate_task_data(task_id=2)
    
    # Step 1: Can we solve Task 2 by just re-routing the binary mask of the frozen network?
    success = evolutionary_mask_search(neuron, X2, y2)
    
    if not success:
        print("\nExisting structure is insufficient. Triggering Structural Allocation...")
        neuron.add_branch()
        
        # Only the new branch and the soma bias should be trainable
        neuron.soma_bias.requires_grad = True
        for param in neuron.branches[-1].parameters():
            param.requires_grad = True
            
        print("Training continuous weights of the NEW structure on Task 2...")
        train_continuous_weights(neuron, X2, y2, epochs=1000)
        
    print("\nDone! Hybrid framework execution complete.")
