import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from auto_mdnn import AutoMDNN

def compute_telemetry(model_name, model, X, y):
    print(f"--- Telemetry for {model_name} ---")
    model.train()
    
    # Forward Pass Breakdown (manually executing for telemetry)
    # 1. Synaptic Layer
    x_expanded = X.unsqueeze(1).unsqueeze(2)
    w_expanded = model.w.unsqueeze(0)
    theta_expanded = model.theta.unsqueeze(0)
    
    pre_act = w_expanded * x_expanded - theta_expanded
    
    if model.syn_func == 0: Y = torch.sigmoid(pre_act)
    elif model.syn_func == 1: Y = torch.sin(pre_act)
    elif model.syn_func == 2: Y = torch.exp(-(pre_act ** 2))
    
    # We want to retain gradients on Y to measure them!
    Y.retain_grad()
    
    # 2. Dendritic Layer
    if model.den_func == 0:
        Z = torch.prod(Y, dim=3)
    elif model.den_func == 1:
        Z, _ = torch.min(Y, dim=3)
    elif model.den_func == 2:
        Z, _ = torch.max(Y, dim=3)
        
    Z.retain_grad()
    
    # Log Activation Statistics
    z_mean = torch.mean(torch.abs(Z)).item()
    z_std = torch.std(Z).item()
    print(f"Dendrite Activation (Z) Mean Abs: {z_mean:.6e}")
    print(f"Dendrite Activation (Z) Std Dev:  {z_std:.6e}")
    
    # 3. Membrane Layer
    if model.mem_func == 0: V = torch.sum(Z, dim=2)
    elif model.mem_func == 1: V, _ = torch.max(Z, dim=2)
    elif model.mem_func == 2: V = torch.mean(Z, dim=2)
    
    # Loss and Backward
    criterion = nn.CrossEntropyLoss()
    loss = criterion(V, y)
    
    model.zero_grad()
    loss.backward()
    
    # Log Gradient Statistics
    grad_norm_Z = torch.norm(Z.grad).item() if Z.grad is not None else 0.0
    grad_norm_W = torch.norm(model.w.grad).item() if model.w.grad is not None else 0.0
    
    print(f"Gradient L2 Norm on Dendrites (Z): {grad_norm_Z:.6e}")
    print(f"Gradient L2 Norm on Weights (W):   {grad_norm_W:.6e}")
    print("\n")

def main():
    data = load_digits()
    X = data.data[:128] # Use a single batch of 128 images
    y = data.target[:128]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    input_dim = X.shape[1]
    num_classes = 10
    
    torch.manual_seed(42)
    
    # 1. Baseline Model: Gaussian -> PRODUCT -> Sum
    baseline_model = AutoMDNN(
        input_dim=input_dim, num_classes=num_classes, num_branches=10,
        syn_func=2, den_func=0, mem_func=0
    )
    
    # 2. Evolved Model: Gaussian -> MIN -> Sum
    evolved_model = AutoMDNN(
        input_dim=input_dim, num_classes=num_classes, num_branches=10,
        syn_func=2, den_func=1, mem_func=0
    )
    
    # Ensure both models start with the EXACT same weights to make comparison perfectly fair!
    evolved_model.w.data = baseline_model.w.data.clone()
    evolved_model.theta.data = baseline_model.theta.data.clone()
    
    print("="*50)
    print("EMPIRICAL TELEMETRY RESULTS")
    print("="*50)
    compute_telemetry("Baseline (PRODUCT Dendrite)", baseline_model, X_tensor, y_tensor)
    compute_telemetry("Evolved (MIN Dendrite)", evolved_model, X_tensor, y_tensor)
    print("="*50)

if __name__ == "__main__":
    main()
