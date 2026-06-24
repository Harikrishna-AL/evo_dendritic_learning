import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auto_mdnn import AutoMDNN

class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        logits_train = model(X_train)
        preds_train = logits_train.argmax(dim=1)
        train_acc = (preds_train == y_train).float().mean().item()
        
        logits_test = model(X_test)
        preds_test = logits_test.argmax(dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
        
    return train_acc, test_acc

def main():
    print("Loading Digits Dataset...")
    data = load_digits()
    X = data.data
    y = data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    input_dim = X.shape[1]
    num_classes = 10
    
    seeds = [42, 101, 2024, 7, 123]
    epochs = 100
    
    print("==================================================")
    print("STEP 1: PARAMETER-MATCHED ADVANTAGE BENCHMARK")
    print("==================================================")
    
    # Model 1: AutoMDNN (Gaussian -> Product -> Sum)
    mdnn = AutoMDNN(input_dim, num_classes, num_branches=10, syn_func=2, den_func=0, mem_func=0)
    print(f"MDNN Parameter Count: {count_parameters(mdnn)}")
    
    # Model 2: Standard MLP
    # To match 12,800 params: (64*H + H) + (H*10 + 10) = 12800 => 76H = 12790 => H ~ 168
    mlp = StandardMLP(input_dim, hidden_dim=168, num_classes=num_classes)
    print(f"Standard MLP Parameter Count: {count_parameters(mlp)}\n")
    
    mdnn_train, mdnn_test = [], []
    mlp_train, mlp_test = [], []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=seed
        )
        
        # Train MDNN
        m = AutoMDNN(input_dim, num_classes, num_branches=10, syn_func=2, den_func=0, mem_func=0)
        tr, te = train_and_evaluate(m, X_train, y_train, X_test, y_test, epochs)
        mdnn_train.append(tr)
        mdnn_test.append(te)
        
        # Train MLP
        m_mlp = StandardMLP(input_dim, 168, num_classes)
        tr, te = train_and_evaluate(m_mlp, X_train, y_train, X_test, y_test, epochs)
        mlp_train.append(tr)
        mlp_test.append(te)
        
    print("--- MDNN (Gaussian Dendritic Neuron) ---")
    print(f"Train: {np.mean(mdnn_train)*100:.2f}% ± {np.std(mdnn_train)*100:.2f}%")
    print(f"Test:  {np.mean(mdnn_test)*100:.2f}% ± {np.std(mdnn_test)*100:.2f}%\n")
    
    print("--- Standard MLP (Normal Neuron) ---")
    print(f"Train: {np.mean(mlp_train)*100:.2f}% ± {np.std(mlp_train)*100:.2f}%")
    print(f"Test:  {np.mean(mlp_test)*100:.2f}% ± {np.std(mlp_test)*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
