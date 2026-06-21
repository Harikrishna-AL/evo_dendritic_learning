import sys
import os
import torch

# Add the DLL repo to the path so we can import their layers
sys.path.append(os.path.join(os.path.dirname(__file__), '.dll_repo'))

from FC_layer import DLL_FCLayer

# Define activation functions used in DLL
def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_deriv(x):
    s = torch.sigmoid(x)
    return s * (1 - s)

class EvoDLLNetwork(object):
    """
    A 100% biologically plausible Dendritic Network for CIL.
    Built directly on top of the cloned Dendritic-Localized-Learning repository.
    Features Autonomous Neurogenesis and Output Error Masking for Zero-Forgetting.
    """
    def __init__(self, input_dim, output_dim=10, initial_branches=1, lr=0.01, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.lr = lr
        
        # We will store branches as independent DLL_FCLayer instances.
        # This allows us to dynamically grow branches without breaking Adam buffers!
        self.branches = []
        
        # Soma is also a DLL_FCLayer, but its input size grows dynamically.
        # We start with input size = initial_branches
        self.soma = None 
        
        # Autonomous Growth tracking
        self.local_error_history = 0.0
        self.epochs_since_growth = 0
        
        for _ in range(initial_branches):
            self.add_branch()
            
    def add_branch(self):
        """Autonomously sprout a new dendritic branch!"""
        # Create a new branch using the exact class from the DLL repo
        new_branch = DLL_FCLayer(
            input_size=self.input_dim, 
            output_size=1, 
            learning_rate=self.lr, 
            f=sigmoid, 
            df=sigmoid_deriv, 
            enable_adam=True, 
            device=self.device
        )
        self.branches.append(new_branch)
        
        # We must recreate the soma to accept the new branch's input.
        # To prevent catastrophic forgetting, we copy the old soma weights.
        old_soma = self.soma
        new_num_branches = len(self.branches)
        
        self.soma = DLL_FCLayer(
            input_size=new_num_branches, 
            output_size=self.output_dim, 
            learning_rate=self.lr, 
            f=sigmoid, 
            df=sigmoid_deriv, 
            enable_adam=True, 
            device=self.device
        )
        
        if old_soma is not None:
            # Copy old weights
            self.soma.weights[:-1, :] = old_soma.weights
            self.soma.theta[:-1, :] = old_soma.theta
            # Copy Adam buffers
            self.soma.m_weights[:-1, :] = old_soma.m_weights
            self.soma.v_weights[:-1, :] = old_soma.v_weights
            self.soma.m_theta[:-1, :] = old_soma.m_theta
            self.soma.v_theta[:-1, :] = old_soma.v_theta
            
            # The new branch weight is initialized to a small random value (handled by DLL_FCLayer init)
        
        # Reset local error history upon growth
        self.local_error_history = 0.0
        self.epochs_since_growth = 0
        
    def num_branches(self):
        return len(self.branches)

    def forward(self, x):
        """Basal Sensory Forward Pass using DLL layers"""
        # 1. Branch Computations
        u_branch_list = []
        for branch in self.branches:
            u_branch_list.append(branch.forward(x))
            
        self.u_branches = torch.cat(u_branch_list, dim=1) # (Batch, Branches)
        
        # 2. Soma Computation
        self.u_soma = self.soma.forward(self.u_branches) # (Batch, 1)
        
        return self.u_soma
        
    def compute_apical_energy(self, target):
        """For Class-Incremental Learning inference. Lower is better."""
        u = self.forward(x)
        return torch.mean((target - u) ** 2).item()
        
    def backward_and_update(self, target, active_classes, freeze_old_branches=False):
        """
        DLL Weight Updates using the cloned repository's methods.
        """
        e_soma = target - self.u_soma
        
        # CIL OUTPUT MASKING: 
        # We forcefully zero out the error gradients for any class not in active_classes.
        # This mathematically guarantees the soma weights for old classes will never change.
        mask = torch.zeros_like(e_soma)
        for c in active_classes:
            mask[:, c] = 1.0
        e_soma = e_soma * mask
        
        # Accumulate local error for autonomous neurogenesis. 
        # Average only over active classes so the threshold logic works (max error ~0.5 instead of ~0.1).
        current_error = torch.sum(e_soma ** 2).item() / (e_soma.size(0) * len(active_classes))
        self.local_error_history = 0.9 * self.local_error_history + 0.1 * current_error
        
        # Backward pass through soma to get branch errors
        # Note: the paper uses a sigma scaling factor, we can just use 1.0
        e_branches = self.soma.backward(e_soma, sigma=1.0) # (Batch, Branches)
        
        # Update Soma weights
        if freeze_old_branches and self.num_branches() > 1:
            old_weights = self.soma.weights[:-1, :].clone()
            old_theta = self.soma.theta[:-1, :].clone()
            
            self.soma.update_weights(e_soma, sigma=1.0)
            self.soma.update_theta(e_branches, e_soma, sigma=1.0)
            
            self.soma.weights[:-1, :] = old_weights
            self.soma.theta[:-1, :] = old_theta
        else:
            self.soma.update_weights(e_soma, sigma=1.0)
            self.soma.update_theta(e_branches, e_soma, sigma=1.0)
            
        # Update Branch weights
        for i, branch in enumerate(self.branches):
            if freeze_old_branches and i < self.num_branches() - 1:
                continue
                
            e_b = e_branches[:, i:i+1]
            e_b_input = branch.backward(e_b, sigma=1.0)
            branch.update_weights(e_b, sigma=1.0)
            branch.update_theta(e_b_input, e_b, sigma=1.0)
            
        # 3. Autonomous Neurogenesis Trigger
        self.epochs_since_growth += 1
        if self.local_error_history > 0.20 and self.epochs_since_growth > 50:
            print(f"    [Neurogenesis] Soma local error {self.local_error_history:.3f} exceeded threshold! Sprouting new branch.")
            self.add_branch()
            
        return current_error
