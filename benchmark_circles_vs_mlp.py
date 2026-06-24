import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_circles
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    X, y = make_circles(n_samples=2000, noise=0.1, factor=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # 2 classes, 5 branches = 2*5*2*2 = 40 parameters
    mdnn = AutoMDNN(2, 2, 5, syn_func=2, den_func=0, mem_func=0)
    print(f"MDNN Parameter Count: {count_parameters(mdnn)}")
    
    # MLP with ~40 parameters => H=8 => (2*8+8) + (8*2+2) = 24 + 18 = 42 parameters
    mlp = StandardMLP(2, 8, 2)
    print(f"MLP Parameter Count: {count_parameters(mlp)}")
    
    seeds = [42, 101, 2024]
    mdnn_accs = []
    mlp_accs = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=seed)
        
        m = AutoMDNN(2, 2, 5, syn_func=2, den_func=0, mem_func=0)
        mdnn_accs.append(train_and_evaluate(m, X_train, y_train, X_test, y_test, 100))
        
        m_mlp = StandardMLP(2, 8, 2)
        mlp_accs.append(train_and_evaluate(m_mlp, X_train, y_train, X_test, y_test, 100))
        
    print(f"MDNN (Gaussian) Test Acc: {np.mean(mdnn_accs)*100:.2f}%")
    print(f"MLP (ReLU) Test Acc:      {np.mean(mlp_accs)*100:.2f}%")

if __name__ == "__main__":
    main()
