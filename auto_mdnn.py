import torch
import torch.nn as nn

class AutoMDNN(nn.Module):
    """
    A Vectorized Multiple Dendritic Neural Network (MDNN) for multi-class tasks.
    It contains N parallel dendritic neurons. The Membrane Layer output is
    returned directly as logits for a global Softmax (CrossEntropyLoss).
    """
    def __init__(self, input_dim, num_classes=10, num_branches=5, 
                 syn_func=0, den_func=0, mem_func=0):
        super(AutoMDNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_branches = num_branches
        
        # 3-Part Architecture DNA (No Soma layer)
        self.syn_func = syn_func
        self.den_func = den_func
        self.mem_func = mem_func
        
        # Vectorized Weights for all classes and branches
        # Shape: (num_classes, num_branches, input_dim)
        self.w = nn.Parameter(torch.randn(num_classes, num_branches, input_dim) * 0.1)
        self.theta = nn.Parameter(torch.randn(num_classes, num_branches, input_dim) * 0.1)

    def forward(self, x):
        """
        x: shape (Batch, input_dim)
        Returns: logits of shape (Batch, num_classes)
        """
        # ---------------------------------------------------------
        # 1. Synaptic Layer
        # x is (B, I), w is (C, Br, I), theta is (C, Br, I)
        # We expand x to (B, 1, 1, I) and w to (1, C, Br, I)
        x_expanded = x.unsqueeze(1).unsqueeze(2) # (B, 1, 1, I)
        w_expanded = self.w.unsqueeze(0)         # (1, C, Br, I)
        theta_expanded = self.theta.unsqueeze(0) # (1, C, Br, I)
        
        # Compute pre-activation for all classes, branches, and features
        pre_act = w_expanded * x_expanded - theta_expanded # Shape: (B, C, Br, I)
        
        # Apply Synaptic Function
        if self.syn_func == 0:
            Y = torch.sigmoid(pre_act)
        elif self.syn_func == 1:
            Y = torch.sin(pre_act)
        elif self.syn_func == 2:
            Y = torch.exp(-(pre_act ** 2))
        else:
            raise ValueError(f"Unknown syn_func: {self.syn_func}")
            
        # ---------------------------------------------------------
        # 2. Dendritic Layer
        # Aggregate across the input dimension (I)
        # Input Y is (B, C, Br, I). Output Z is (B, C, Br)
        if self.den_func == 0:
            Z = torch.prod(Y, dim=3)
        elif self.den_func == 1:
            Z, _ = torch.min(Y, dim=3)
        elif self.den_func == 2:
            Z, _ = torch.max(Y, dim=3)
        else:
            raise ValueError(f"Unknown den_func: {self.den_func}")
            
        # ---------------------------------------------------------
        # 3. Membrane Layer
        # Aggregate across the branches (Br)
        # Input Z is (B, C, Br). Output V is (B, C)
        if self.mem_func == 0:
            V = torch.sum(Z, dim=2)
        elif self.mem_func == 1:
            V, _ = torch.max(Z, dim=2)
        elif self.mem_func == 2:
            V = torch.mean(Z, dim=2)
        else:
            raise ValueError(f"Unknown mem_func: {self.mem_func}")
            
        # V represents the raw Membrane Voltages for all N classes
        # This is returned directly as Logits for CrossEntropyLoss/Softmax
        return V

    def get_architecture_string(self):
        syn_names = ["Sigmoid", "Sine", "Gaussian"]
        den_names = ["Product (AND)", "Min (Fuzzy AND)", "Max (Pool)"]
        mem_names = ["Sum (OR)", "Max (WTA)", "Mean (Avg)"]
        
        return (f"Synapse: {syn_names[self.syn_func]} -> "
                f"Dendrite: {den_names[self.den_func]} -> "
                f"Membrane: {mem_names[self.mem_func]} -> "
                f"Soma: (Bypassed to Softmax)")
