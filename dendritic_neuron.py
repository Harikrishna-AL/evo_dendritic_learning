import torch
import torch.nn as nn
import torch.nn.functional as F

class DendriticBranch(nn.Module):
    r"""
    A single dendritic branch.
    Computes a linear sum of its inputs and applies a static nonlinearity.
    b( \sum_{i=1}^k w_{ij} x_{ij} )
    """
    def __init__(self, input_dim):
        super().__init__()
        # In a biological setting, this might be sparse (k << input_dim).
        # For our initial PyTorch model, we use a full linear layer 
        # and will enforce sparsity/routing later.
        self.linear = nn.Linear(input_dim, 1, bias=False)
        # The weight of this branch's connection to the soma (can be negative for inhibition)
        self.branch_weight = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        # Apply linear sum, static nonlinearity, and the branch weight
        return self.branch_weight * torch.sigmoid(self.linear(x))

class DendriticNeuron(nn.Module):
    r"""
    A neuron composed of multiple dendritic branches.
    a_N(x) = \sum_{j=1}^m b( \sum_{i=1}^k w_{ij} x_{ij} )
    """
    def __init__(self, input_dim, initial_branches=1):
        super().__init__()
        self.input_dim = input_dim
        # We store branches in a ModuleList so we can dynamically add more
        self.branches = nn.ModuleList([
            DendriticBranch(input_dim) for _ in range(initial_branches)
        ])
        # The soma needs a bias (threshold) to map the sum of branch outputs back to [0, 1]
        self.soma_bias = nn.Parameter(torch.zeros(1))
        
    def add_branch(self):
        """Dynamically add a new branch to the neuron's structure."""
        new_branch = DendriticBranch(self.input_dim)
        # Move to the same device as existing branches
        if len(self.branches) > 0:
            device = next(self.branches[0].parameters()).device
            new_branch.to(device)
            
        self.branches.append(new_branch)
        print(f"Neuron grew a new branch! Total branches: {len(self.branches)}")
        
    def remove_branch(self, index):
        """Remove a specific branch (for evolutionary pruning)."""
        if 0 <= index < len(self.branches):
            del self.branches[index]
            print(f"Branch {index} pruned. Total branches: {len(self.branches)}")
            
    def forward(self, x):
        r"""
        The output is the linear sum of all branch outputs, passed through a final soma activation.
        a_{soma} = \sigma( \sum a_{branch} + bias )
        """
        if len(self.branches) == 0:
            return torch.zeros(x.size(0), 1, device=x.device)
            
        # Compute output for each branch: shape (batch_size, 1)
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Sum them up and apply soma threshold/bias
        stacked = torch.stack(branch_outputs, dim=1) # (batch, m, 1)
        dendritic_sum = torch.sum(stacked, dim=1) # (batch, 1)
        return torch.sigmoid(dendritic_sum + self.soma_bias)

# Quick test
if __name__ == "__main__":
    # Create a neuron with 2 branches initially
    neuron = DendriticNeuron(input_dim=10, initial_branches=2)
    print(neuron)
    
    # Dummy input
    x = torch.randn(5, 10) # Batch size 5, input dim 10
    out = neuron(x)
    print("Output shape:", out.shape)
    
    # Simulate structural growth
    neuron.add_branch()
    out = neuron(x)
    print("Output shape after growth:", out.shape)
