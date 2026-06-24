import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dendritic_layer import DendriticLayer

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    
    print("==================================================")
    print("DEEP DENDRITIC NEURAL NETWORK (DDNN) TEST")
    print("==================================================")
    
    # 1. Shallow MDNN (1 Layer)
    model_shallow = nn.Sequential(
        DendriticLayer(in_features=input_dim, out_features=num_classes, num_branches=10)
    )
    print(f"Shallow DDNN Parameters: {count_parameters(model_shallow)}")
    print(model_shallow)
    
    # 2. Deep MDNN (2 Layers) 
    # Note: No ReLU needed between layers because DendriticLayer is inherently non-linear!
    model_deep = nn.Sequential(
        DendriticLayer(in_features=input_dim, out_features=64, num_branches=5),
        DendriticLayer(in_features=64, out_features=num_classes, num_branches=5)
    )
    print(f"\nDeep DDNN (2-Layers) Parameters: {count_parameters(model_deep)}")
    print(model_deep)
    print("\nTraining models over 3 seeds to confirm gradient flow and capacity...\n")
    
    seeds = [42, 101, 2024]
    shallow_test = []
    deep_test = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=seed
        )
        
        m_shallow = nn.Sequential(
            DendriticLayer(input_dim, num_classes, num_branches=10)
        )
        tr, te = train_and_evaluate(m_shallow, X_train, y_train, X_test, y_test, epochs=100)
        shallow_test.append(te)
        
        m_deep = nn.Sequential(
            DendriticLayer(input_dim, 64, num_branches=5),
            DendriticLayer(64, num_classes, num_branches=5)
        )
        tr, te = train_and_evaluate(m_deep, X_train, y_train, X_test, y_test, epochs=100)
        deep_test.append(te)
        
    print("--- Shallow DDNN (1 Layer) ---")
    print(f"Test Accuracy: {np.mean(shallow_test)*100:.2f}% ± {np.std(shallow_test)*100:.2f}%\n")
    
    print("--- Deep DDNN (2 Stacked Layers) ---")
    print(f"Test Accuracy: {np.mean(deep_test)*100:.2f}% ± {np.std(deep_test)*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
