import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auto_mdnn import AutoMDNN

def train_and_evaluate(model, X_train, y_train, X_test, y_test, lr, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        logits_test = model(X_test)
        preds_test = logits_test.argmax(dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
        
    return test_acc

def main():
    print("Loading Digits Dataset for LR Sweep...")
    data = load_digits()
    X = data.data
    y = data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    input_dim = X.shape[1]
    num_classes = 10
    
    seeds = [42, 101, 2024]
    lrs = [0.001, 0.01, 0.05, 0.1]
    epochs = 100
    
    print("==================================================")
    print("LEARNING RATE SWEEP (SIGMOID -> PRODUCT -> SUM)")
    print("==================================================")
    
    for lr in lrs:
        test_accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=seed
            )
            
            # True Biological Baseline: Sigmoid (0) -> Product (0) -> Sum (0)
            model = AutoMDNN(
                input_dim=input_dim, num_classes=num_classes, num_branches=10,
                syn_func=0, den_func=0, mem_func=0
            )
            
            acc = train_and_evaluate(model, X_train, y_train, X_test, y_test, lr=lr, epochs=epochs)
            test_accs.append(acc)
            
        print(f"LR {lr}: Test Acc = {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")

if __name__ == "__main__":
    main()
