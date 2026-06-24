import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auto_mdnn import AutoMDNN

def make_checkerboard(n_samples=2000, noise=0.1):
    X = np.random.uniform(-4, 4, size=(n_samples, 2))
    y = (np.sin(X[:, 0] * 1.5) * np.sin(X[:, 1] * 1.5) > 0).astype(int)
    X += np.random.normal(0, noise, size=X.shape)
    return X, y

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
        logits_test = model(X_test)
        preds_test = logits_test.argmax(dim=1)
        test_acc = (preds_test == y_test).float().mean().item()
        
    return test_acc

def main():
    X, y = make_checkerboard(n_samples=2000, noise=0.1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Force Sine Architecture
    model_sine = AutoMDNN(2, 2, 5, syn_func=1, den_func=1, mem_func=1)
    acc_sine = train_and_evaluate(model_sine, X_train, y_train, X_test, y_test, epochs=200)
    print(f"Forced Sine Accuracy: {acc_sine*100:.2f}%")

if __name__ == "__main__":
    main()
