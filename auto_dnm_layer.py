import torch
import torch.nn as nn
import math

class AutoDNMLayer(nn.Module):
    """
    An AutoML Dendritic Neuron Model where the mathematical functions of each
    of the 4 biological layers can be dynamically swapped using integer flags.
    """
    def __init__(self, input_dim, num_branches=5, 
                 syn_func=0, den_func=0, mem_func=0, soma_func=0):
        super(AutoDNMLayer, self).__init__()
        
        self.input_dim = input_dim
        self.num_branches = num_branches
        
        # Architecture DNA
        self.syn_func = syn_func
        self.den_func = den_func
        self.mem_func = mem_func
        self.soma_func = soma_func
        
        # Learnable Weights (Weight DNA)
        # Shape: (num_branches, input_dim)
        self.w = nn.Parameter(torch.randn(num_branches, input_dim) * 0.1)
        self.theta = nn.Parameter(torch.randn(num_branches, input_dim) * 0.1)
        
        # Soma Parameters
        self.k_s = nn.Parameter(torch.tensor(1.0))
        self.theta_s = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        x: shape (Batch, input_dim)
        """
        batch_size = x.size(0)
        
        # ---------------------------------------------------------
        # 1. Synaptic Layer
        # Compute w * x - theta for all branches simultaneously
        # x is (B, I), w is (Br, I), theta is (Br, I)
        # We want pre_activation of shape (B, Br, I)
        x_expanded = x.unsqueeze(1) # (B, 1, I)
        pre_act = self.w.unsqueeze(0) * x_expanded - self.theta.unsqueeze(0) # (B, Br, I)
        
        # Apply Synaptic Function
        if self.syn_func == 0:
            # Sigmoid (Pass/Prune)
            Y = torch.sigmoid(pre_act)
        elif self.syn_func == 1:
            # Sine (Periodic)
            Y = torch.sin(pre_act)
        elif self.syn_func == 2:
            # Gaussian (Radial Basis)
            Y = torch.exp(-(pre_act ** 2))
        else:
            raise ValueError(f"Unknown syn_func: {self.syn_func}")
            
        # ---------------------------------------------------------
        # 2. Dendritic Layer
        # Aggregate across the input dimension (I) for each branch
        # Input Y is (B, Br, I). Output Z is (B, Br)
        if self.den_func == 0:
            # Product (Boolean AND)
            Z = torch.prod(Y, dim=2)
        elif self.den_func == 1:
            # Min (Fuzzy AND)
            Z, _ = torch.min(Y, dim=2)
        elif self.den_func == 2:
            # Max (Local Pooling)
            Z, _ = torch.max(Y, dim=2)
        else:
            raise ValueError(f"Unknown den_func: {self.den_func}")
            
        # ---------------------------------------------------------
        # 3. Membrane Layer
        # Aggregate across the branches (Br)
        # Input Z is (B, Br). Output V is (B)
        if self.mem_func == 0:
            # Sum (Boolean OR)
            V = torch.sum(Z, dim=1)
        elif self.mem_func == 1:
            # Max (Winner-Takes-All)
            V, _ = torch.max(Z, dim=1)
        elif self.mem_func == 2:
            # Mean (Averaging)
            V = torch.mean(Z, dim=1)
        else:
            raise ValueError(f"Unknown mem_func: {self.mem_func}")
            
        # ---------------------------------------------------------
        # 4. Soma Layer
        # Final activation
        # Input V is (B). Output O is (B, 1)
        soma_pre = self.k_s * V - self.theta_s
        
        if self.soma_func == 0:
            # Sigmoid (Soft spike)
            O = torch.sigmoid(soma_pre)
        elif self.soma_func == 1:
            # Step (Hard spike) approximated with steep sigmoid for backprop
            O = torch.sigmoid(soma_pre * 10.0)
        elif self.soma_func == 2:
            # Tanh (Signed output)
            O = torch.tanh(soma_pre)
        else:
            raise ValueError(f"Unknown soma_func: {self.soma_func}")
            
        return O.unsqueeze(1) # Shape: (B, 1)

    def get_architecture_string(self):
        syn_names = ["Sigmoid", "Sine", "Gaussian"]
        den_names = ["Product (AND)", "Min (Fuzzy AND)", "Max (Pool)"]
        mem_names = ["Sum (OR)", "Max (WTA)", "Mean (Avg)"]
        soma_names = ["Sigmoid", "Step", "Tanh"]
        
        return (f"Synapse: {syn_names[self.syn_func]} -> "
                f"Dendrite: {den_names[self.den_func]} -> "
                f"Membrane: {mem_names[self.mem_func]} -> "
                f"Soma: {soma_names[self.soma_func]}")
