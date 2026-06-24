import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auto_mdnn import AutoMDNN

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=50):
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
        # Test Acc
        logits_test = model(X_test)
        preds_test = logits_test.argmax(dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
        
    return test_acc

def main():
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
    
    true_baseline_test = []
    nas_discovered_test = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=seed
        )
        
        # TRUE Biological Baseline: Sigmoid (0) -> Product (0) -> Sum (0)
        true_baseline = AutoMDNN(
            input_dim=input_dim, num_classes=num_classes, num_branches=10,
            syn_func=0, den_func=0, mem_func=0
        )
        te_acc1 = train_and_evaluate(true_baseline, X_train, y_train, X_test, y_test, epochs=epochs)
        true_baseline_test.append(te_acc1)
        
        # NAS Discovered Optimal: Gaussian (2) -> Product (0) -> Sum (0)
        # (This is what NAS converged to if we discount the fast-learning Min illusion)
        nas_discovered = AutoMDNN(
            input_dim=input_dim, num_classes=num_classes, num_branches=10,
            syn_func=2, den_func=0, mem_func=0
        )
        te_acc2 = train_and_evaluate(nas_discovered, X_train, y_train, X_test, y_test, epochs=epochs)
        nas_discovered_test.append(te_acc2)
        
    print(f"True Biological Baseline (SIGMOID): {np.mean(true_baseline_test)*100:.2f}% ± {np.std(true_baseline_test)*100:.2f}%")
    print(f"NAS Discovered Architecture (GAUSSIAN): {np.mean(nas_discovered_test)*100:.2f}% ± {np.std(nas_discovered_test)*100:.2f}%")

if __name__ == "__main__":
    main()
