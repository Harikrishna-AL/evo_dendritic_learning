import torch
import torch.nn as nn

class DendriticLayer(nn.Module):
    """
    A stackable, modular Dendritic Neural Layer for PyTorch.
    Acts as a drop-in replacement for nn.Linear, but utilizes the evolved
    biological math: Synapse -> Dendrite -> Membrane.
    """
    def __init__(self, in_features, out_features, num_branches=5, 
                 syn_func=2, den_func=0, mem_func=0):
        super(DendriticLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_branches = num_branches
        
        # 3-Part Architecture DNA
        self.syn_func = syn_func
        self.den_func = den_func
        self.mem_func = mem_func
        
        # Vectorized Weights
        # Shape: (out_features, num_branches, in_features)
        self.w = nn.Parameter(torch.randn(out_features, num_branches, in_features) * 0.1)
        self.theta = nn.Parameter(torch.randn(out_features, num_branches, in_features) * 0.1)

    def forward(self, x):
        """
        x: shape (Batch, in_features)
        Returns: shape (Batch, out_features)
        """
        # 1. Synaptic Layer
        # x is (B, I), w is (O, Br, I), theta is (O, Br, I)
        x_expanded = x.unsqueeze(1).unsqueeze(2) # (B, 1, 1, I)
        w_expanded = self.w.unsqueeze(0)         # (1, O, Br, I)
        theta_expanded = self.theta.unsqueeze(0) # (1, O, Br, I)
        
        pre_act = w_expanded * x_expanded - theta_expanded # (B, O, Br, I)
        
        if self.syn_func == 0: Y = torch.sigmoid(pre_act)
        elif self.syn_func == 1: Y = torch.sin(pre_act)
        elif self.syn_func == 2: Y = torch.exp(-(pre_act ** 2))
        else: raise ValueError(f"Unknown syn_func: {self.syn_func}")
            
        # 2. Dendritic Layer
        # Aggregate across the input dimension (I) -> (B, O, Br)
        if self.den_func == 0: Z = torch.prod(Y, dim=3)
        elif self.den_func == 1: Z, _ = torch.min(Y, dim=3)
        elif self.den_func == 2: Z, _ = torch.max(Y, dim=3)
        else: raise ValueError(f"Unknown den_func: {self.den_func}")
            
        # 3. Membrane Layer
        # Aggregate across the branches (Br) -> (B, O)
        if self.mem_func == 0: V = torch.sum(Z, dim=2)
        elif self.mem_func == 1: V, _ = torch.max(Z, dim=2)
        elif self.mem_func == 2: V = torch.mean(Z, dim=2)
        else: raise ValueError(f"Unknown mem_func: {self.mem_func}")
            
        return V

    def extra_repr(self) -> str:
        syn_names = ["Sigmoid", "Sine", "Gaussian"]
        den_names = ["Product (AND)", "Min (Fuzzy AND)", "Max (Pool)"]
        mem_names = ["Sum (OR)", "Max (WTA)", "Mean (Avg)"]
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'branches={self.num_branches}, syn={syn_names[self.syn_func]}, '
                f'den={den_names[self.den_func]}, mem={mem_names[self.mem_func]}')
