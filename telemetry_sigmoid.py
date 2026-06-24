import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from auto_mdnn import AutoMDNN

def compute_telemetry(model_name, model, X, y):
    print(f"--- Telemetry for {model_name} ---")
    model.train()
    
    # Forward Pass Breakdown
    x_expanded = X.unsqueeze(1).unsqueeze(2)
    w_expanded = model.w.unsqueeze(0)
    theta_expanded = model.theta.unsqueeze(0)
    
    pre_act = w_expanded * x_expanded - theta_expanded
    
    # 1. Synaptic Layer
    if model.syn_func == 0: Y = torch.sigmoid(pre_act)
    elif model.syn_func == 1: Y = torch.sin(pre_act)
    elif model.syn_func == 2: Y = torch.exp(-(pre_act ** 2))
    
    Y.retain_grad()
    
    # 2. Dendritic Layer
    if model.den_func == 0: Z = torch.prod(Y, dim=3)
    elif model.den_func == 1: Z, _ = torch.min(Y, dim=3)
    elif model.den_func == 2: Z, _ = torch.max(Y, dim=3)
        
    Z.retain_grad()
    
    # 3. Membrane Layer
    if model.mem_func == 0: V = torch.sum(Z, dim=2)
    elif model.mem_func == 1: V, _ = torch.max(Z, dim=2)
    elif model.mem_func == 2: V = torch.mean(Z, dim=2)
    
    # Loss and Backward
    criterion = nn.CrossEntropyLoss()
    loss = criterion(V, y)
    
    model.zero_grad()
    loss.backward()
    
    # Log Statistics
    y_mean = torch.mean(Y).item()
    z_mean = torch.mean(Z).item()
    
    grad_norm_Y = torch.norm(Y.grad).item() if Y.grad is not None else 0.0
    grad_norm_W = torch.norm(model.w.grad).item() if model.w.grad is not None else 0.0
    
    print(f"Synapse Activation (Y) Mean: {y_mean:.6e}")
    print(f"Dendrite Activation (Z) Mean: {z_mean:.6e}")
    print(f"Gradient Norm at Synapse Output (Y): {grad_norm_Y:.6e}")
    print(f"Gradient Norm at Weights (W): {grad_norm_W:.6e}")
    print("\n")

def main():
    data = load_digits()
    X = data.data[:128] # Single batch
    y = data.target[:128]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    input_dim = X.shape[1]
    num_classes = 10
    
    torch.manual_seed(42)
    
    # 1. Biological Model: SIGMOID -> PRODUCT
    sigmoid_model = AutoMDNN(
        input_dim=input_dim, num_classes=num_classes, num_branches=10,
        syn_func=0, den_func=0, mem_func=0
    )
    
    # 2. RBF Model: GAUSSIAN -> PRODUCT
    gaussian_model = AutoMDNN(
        input_dim=input_dim, num_classes=num_classes, num_branches=10,
        syn_func=2, den_func=0, mem_func=0
    )
    
    # Exact same initialization
    gaussian_model.w.data = sigmoid_model.w.data.clone()
    gaussian_model.theta.data = sigmoid_model.theta.data.clone()
    
    print("==================================================")
    print("SYNAPTIC GRADIENT TELEMETRY")
    print("==================================================")
    compute_telemetry("Biological Baseline (SIGMOID+PRODUCT)", sigmoid_model, X_tensor, y_tensor)
    compute_telemetry("RBF Architecture (GAUSSIAN+PRODUCT)", gaussian_model, X_tensor, y_tensor)
    print("==================================================")

if __name__ == "__main__":
    main()
