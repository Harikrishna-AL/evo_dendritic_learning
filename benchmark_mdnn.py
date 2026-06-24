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
        # Train Acc
        logits_train = model(X_train)
        preds_train = logits_train.argmax(dim=1)
        train_acc = (preds_train == y_train).float().mean().item()
        
        # Test Acc
        logits_test = model(X_test)
        preds_test = logits_test.argmax(dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
        
    return train_acc, test_acc

def main():
    print("Loading Digits Dataset for Benchmarking...")
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
    
    print(f"Running Multi-Seed Benchmark (N={len(seeds)}, Epochs={epochs})...")
    
    baseline_train = []
    baseline_test = []
    evolved_train = []
    evolved_test = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=seed
        )
        
        # 1. Baseline Model: Gaussian -> PRODUCT -> Sum
        baseline_model = AutoMDNN(
            input_dim=input_dim, num_classes=num_classes, num_branches=10,
            syn_func=2, den_func=0, mem_func=0
        )
        tr_acc, te_acc = train_and_evaluate(baseline_model, X_train, y_train, X_test, y_test, epochs=epochs)
        baseline_train.append(tr_acc)
        baseline_test.append(te_acc)
        
        # 2. Evolved Model: Gaussian -> MIN -> Sum
        evolved_model = AutoMDNN(
            input_dim=input_dim, num_classes=num_classes, num_branches=10,
            syn_func=2, den_func=1, mem_func=0
        )
        tr_acc, te_acc = train_and_evaluate(evolved_model, X_train, y_train, X_test, y_test, epochs=epochs)
        evolved_train.append(tr_acc)
        evolved_test.append(te_acc)
        
    print("\n" + "="*50)
    print("STATISTICAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Baseline (PRODUCT Dendrite):")
    print(f"  Train Acc: {np.mean(baseline_train)*100:.2f}% ± {np.std(baseline_train)*100:.2f}%")
    print(f"  Test Acc:  {np.mean(baseline_test)*100:.2f}% ± {np.std(baseline_test)*100:.2f}%")
    print()
    print(f"Evolved (MIN Dendrite):")
    print(f"  Train Acc: {np.mean(evolved_train)*100:.2f}% ± {np.std(evolved_train)*100:.2f}%")
    print(f"  Test Acc:  {np.mean(evolved_test)*100:.2f}% ± {np.std(evolved_test)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
