import torch
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.dll_repo'))
from FC_layer import DLL_FCLayer

def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_deriv(x):
    s = torch.sigmoid(x)
    return s * (1 - s)

def linear_deriv(x):
    return torch.ones_like(x)

def softmax(x):
    return F.softmax(x, dim=-1)

class DynamicUnifiedDLL(object):
    def __init__(self, input_dim, output_dim, initial_hidden=64, lr=0.01, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.lr = lr
        
        self.frozen_H = 0
        self.current_H = initial_hidden
        self.frozen_classes = []
        
        # 3-layer unified network
        self.layer1 = DLL_FCLayer(input_dim, initial_hidden, learning_rate=lr, f=sigmoid, df=sigmoid_deriv, enable_adam=True, device=device)
        self.layer2 = DLL_FCLayer(initial_hidden, initial_hidden, learning_rate=lr, f=sigmoid, df=sigmoid_deriv, enable_adam=True, device=device)
        self.layer3 = DLL_FCLayer(initial_hidden, output_dim, learning_rate=lr, f=softmax, df=linear_deriv, enable_adam=True, device=device)

    def apply_kwta(self, u, k_percent=0.15):
        """
        Soft k-Winner-Takes-All Lateral Inhibition.
        Hyperpolarizes losing neurons to -50.0 so they output exactly 0.0 and receive 0.0 gradients.
        """
        if k_percent >= 1.0:
            return u
            
        # Determine k dynamically based on current hidden size
        k = max(1, int(u.size(1) * k_percent))
        
        # Get the threshold value (k-th largest activation in each batch item)
        kth_values = torch.topk(u, k, dim=1)[0][:, -1:] 
        
        # Excitatory mask
        mask = (u >= kth_values).float()
        
        # Hyperpolarize the inhibited neurons
        u_inhibited = u * mask + (-50.0) * (1.0 - mask)
        return u_inhibited

    def forward(self, x):
        u1 = self.layer1.forward(x)
        u1_inhibited = self.apply_kwta(u1, k_percent=0.15)
        
        u2 = self.layer2.forward(u1_inhibited)
        u2_inhibited = self.apply_kwta(u2, k_percent=0.15)
        
        out = self.layer3.forward(u2_inhibited)
        return out

    def _expand_matrix_2d(self, mat, dim_add_0=0, dim_add_1=0, init_zero=False):
        """Helper to expand a 2D matrix natively."""
        s0, s1 = mat.shape
        new_mat = torch.empty([s0 + dim_add_0, s1 + dim_add_1], dtype=mat.dtype, device=mat.device)
        if init_zero:
            new_mat.fill_(0.0)
        else:
            new_mat.normal_(mean=0.0, std=0.05)
            
        new_mat[:s0, :s1] = mat
        return new_mat

    def expand(self, num_new_neurons):
        """
        Dynamically sprouts new neurons in the hidden layers while preserving all old connections!
        """
        if num_new_neurons == 0:
            return
            
        N = num_new_neurons
        H = self.current_H
        
        # ---------------- Layer 1: [784, H] -> [784, H+N] ----------------
        self.layer1.weights = self._expand_matrix_2d(self.layer1.weights, 0, N)
        self.layer1.theta = self._expand_matrix_2d(self.layer1.theta, 0, N)
        self.layer1.m_weights = self._expand_matrix_2d(self.layer1.m_weights, 0, N, init_zero=True)
        self.layer1.v_weights = self._expand_matrix_2d(self.layer1.v_weights, 0, N, init_zero=True)
        self.layer1.m_theta = self._expand_matrix_2d(self.layer1.m_theta, 0, N, init_zero=True)
        self.layer1.v_theta = self._expand_matrix_2d(self.layer1.v_theta, 0, N, init_zero=True)
        self.layer1.output_size += N
        
        # ---------------- Layer 2: [H, H] -> [H+N, H+N] ----------------
        # w2[H:, :H] MUST be 0.0 to prevent NEW neurons from injecting into OLD neurons.
        new_w2 = self._expand_matrix_2d(self.layer2.weights, N, N)
        new_w2[H:, :H] = 0.0 # Strict progressive isolation
        self.layer2.weights = new_w2
        
        new_theta2 = self._expand_matrix_2d(self.layer2.theta, N, N)
        new_theta2[H:, :H] = 0.0 
        self.layer2.theta = new_theta2
        
        self.layer2.m_weights = self._expand_matrix_2d(self.layer2.m_weights, N, N, init_zero=True)
        self.layer2.v_weights = self._expand_matrix_2d(self.layer2.v_weights, N, N, init_zero=True)
        self.layer2.m_theta = self._expand_matrix_2d(self.layer2.m_theta, N, N, init_zero=True)
        self.layer2.v_theta = self._expand_matrix_2d(self.layer2.v_theta, N, N, init_zero=True)
        self.layer2.input_size += N
        self.layer2.output_size += N
        
        # ---------------- Layer 3: [H, 10] -> [H+N, 10] ----------------
        new_w3 = self._expand_matrix_2d(self.layer3.weights, N, 0)
        # Prevent new features from modifying outputs of old classes
        for c in self.frozen_classes:
            new_w3[H:, c] = 0.0
        self.layer3.weights = new_w3
        
        new_theta3 = self._expand_matrix_2d(self.layer3.theta, N, 0)
        for c in self.frozen_classes:
            new_theta3[H:, c] = 0.0
        self.layer3.theta = new_theta3
        
        self.layer3.m_weights = self._expand_matrix_2d(self.layer3.m_weights, N, 0, init_zero=True)
        self.layer3.v_weights = self._expand_matrix_2d(self.layer3.v_weights, N, 0, init_zero=True)
        self.layer3.m_theta = self._expand_matrix_2d(self.layer3.m_theta, N, 0, init_zero=True)
        self.layer3.v_theta = self._expand_matrix_2d(self.layer3.v_theta, N, 0, init_zero=True)
        self.layer3.input_size += N
        
        self.current_H += N

    def freeze_current_state(self, active_classes):
        """
        Marks current hidden neurons as permanently frozen.
        Also marks the active_classes as finished, so new neurons won't mess with their output logits.
        """
        self.frozen_H = self.current_H
        for c in active_classes:
            if c not in self.frozen_classes:
                self.frozen_classes.append(c)

    def backward_and_update(self, x, target, active_classes):
        """
        Runs local DLL error propagation.
        Strictly restores the weights of frozen neurons to absolutely prevent Catastrophic Forgetting.
        """
        # Forward pass
        out = self.forward(x)
        
        # Cross Entropy error derivative: Target - Softmax
        e3 = target - out
        
        # Backprop error locally
        e2 = self.layer3.backward(e3, sigma=1.0)
        e1 = self.layer2.backward(e2, sigma=1.0)
        
        # ---------------- Preserve Frozen Weights ----------------
        old_w1 = None
        old_theta1 = None
        old_w2 = None
        old_theta2 = None
        old_w3 = None
        old_theta3 = None
        
        H_f = self.frozen_H
        if H_f > 0:
            old_w1 = self.layer1.weights[:, :H_f].clone()
            old_theta1 = self.layer1.theta[:, :H_f].clone()
            
            # Layer 2 column 0:H_f is completely frozen
            old_w2 = self.layer2.weights[:, :H_f].clone()
            old_theta2 = self.layer2.theta[:, :H_f].clone()
            
            # Layer 3 outputs for frozen classes are frozen
            old_w3_list = [self.layer3.weights[:, c].clone() for c in self.frozen_classes]
            old_theta3_list = [self.layer3.theta[:, c].clone() for c in self.frozen_classes]

        # ---------------- DLL Local Updates ----------------
        self.layer3.update_weights(e3, sigma=1.0)
        self.layer3.update_theta(e2, e3, sigma=1.0)
        
        self.layer2.update_weights(e2, sigma=1.0)
        self.layer2.update_theta(e1, e2, sigma=1.0)
        
        self.layer1.update_weights(e1, sigma=1.0)
        # Layer 1 is connected to inputs, doesn't need to propagate theta further backwards
        
        # ---------------- Restore Frozen Weights ----------------
        if H_f > 0:
            self.layer1.weights[:, :H_f] = old_w1
            self.layer1.theta[:, :H_f] = old_theta1
            
            self.layer2.weights[:, :H_f] = old_w2
            self.layer2.theta[:, :H_f] = old_theta2
            # Strictly enforce zeroing out NEW -> OLD connections, just in case Adam accumulated momentum
            self.layer2.weights[H_f:, :H_f] = 0.0
            self.layer2.theta[H_f:, :H_f] = 0.0
            
            for i, c in enumerate(self.frozen_classes):
                self.layer3.weights[:, c] = old_w3_list[i]
                self.layer3.theta[:, c] = old_theta3_list[i]
                # Enforce zeroing out NEW -> OLD CLASS connections
                self.layer3.weights[H_f:, c] = 0.0
                self.layer3.theta[H_f:, c] = 0.0

        # Calculate tracking loss (Cross Entropy)
        return F.cross_entropy(out, target).item()
