import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from evo_dll_network import EvoDLLNetwork, sigmoid, sigmoid_deriv

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.dll_repo'))
from FC_layer import DLL_FCLayer

device = "cpu"

def get_full_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def get_split_mnist_tasks():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    tasks = []
    for t in range(5):
        classes = [2*t, 2*t + 1]
        train_indices = [i for i, label in enumerate(train_dataset.targets) if label in classes]
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
        
        tasks.append({
            'classes': classes,
            'train_loader': DataLoader(Subset(train_dataset, train_indices), batch_size=256, shuffle=True),
            'test_loader': DataLoader(Subset(test_dataset, test_indices), batch_size=1000, shuffle=False)
        })
    return tasks

class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 1. DLL OFFLINE TRAINING (Paper Replication)
# ==============================================================================
def linear_deriv(x):
    return torch.ones_like(x)

def softmax(x):
    return F.softmax(x, dim=-1)

def run_dll_offline_paper():
    print("\n" + "="*50)
    print("1. DLL OFFLINE TRAINING (As Done in the Paper)")
    print("="*50)
    train_loader, test_loader = get_full_mnist()
    
    # 784 -> 128 -> 128 -> 10 Unified Network
    layer1 = DLL_FCLayer(784, 128, learning_rate=0.001, f=sigmoid, df=sigmoid_deriv, enable_adam=True, device=device)
    layer2 = DLL_FCLayer(128, 128, learning_rate=0.001, f=sigmoid, df=sigmoid_deriv, enable_adam=True, device=device)
    layer3 = DLL_FCLayer(128, 10, learning_rate=0.001, f=softmax, df=linear_deriv, enable_adam=True, device=device)
    
    for epoch in range(10): 
        total_loss = 0.0
        batches = 0
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target_onehot = torch.zeros(target.size(0), 10).to(device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            
            # Forward pass
            x1 = layer1.forward(data)
            x2 = layer2.forward(x1)
            out = layer3.forward(x2) # Probs
            
            # Loss for tracking
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            batches += 1
            
            # Backward pass (Target - Probs is exactly -cross_entropy_deriv)
            e3 = target_onehot - out
            e2 = layer3.backward(e3, sigma=1.0)
            e1 = layer2.backward(e2, sigma=1.0)
            
            # Update weights
            layer3.update_weights(e3, sigma=1.0)
            layer3.update_theta(e2, e3, sigma=1.0)
            
            layer2.update_weights(e2, sigma=1.0)
            layer2.update_theta(e1, e2, sigma=1.0)
            
            layer1.update_weights(e1, sigma=1.0)
            # layer1 is input layer, doesn't need to propagate theta backward
            
        print(f"  Epoch {epoch+1}/10 | Loss: {total_loss/batches:.4f}")
        
    # Evaluate
    correct = 0
    total = 0
    for data, target in test_loader:
        data = data.view(data.size(0), -1).to(device)
        x1 = layer1.forward(data)
        x2 = layer2.forward(x1)
        out = layer3.forward(x2)
        pred = out.argmax(dim=1)
        correct += (pred == target.to(device)).sum().item()
        total += target.size(0)
        
    print(f"---> DLL Offline Accuracy (Paper Style): {100.0 * correct / total:.2f}%\n")

# ==============================================================================
# 1b. EvoDLL OFFLINE TRAINING (Independent Networks / Imbalanced)
# ==============================================================================
def run_evodll_offline():
    print("\n" + "="*50)
    print("1b. EvoDLL OFFLINE TRAINING (10 Independent MSE Networks)")
    print("="*50)
    train_loader, test_loader = get_full_mnist()
    
    # 10 Independent Networks. We give them 5 initial branches because a 1-vs-9 task is much harder than a 1-vs-1 task, 
    # and the diluted error (0.09) won't immediately trigger neurogenesis (threshold 0.20).
    models = [EvoDLLNetwork(input_dim=784, output_dim=1, initial_branches=5, lr=0.01) for _ in range(10)]
    
    for epoch in range(10): # 10 epochs for offline
        total_loss = 0.0
        batches = 0
        for data, target in train_loader:
            data = data.view(data.size(0), -1)
            batch_loss = 0.0
            
            # Train all 10 networks simultaneously on all data
            for c in range(10):
                y_c = (target == c).float().unsqueeze(1)
                model_c = models[c]
                out_c = model_c.forward(data)
                loss_c = model_c.backward_and_update(y_c, active_classes=[0], freeze_old_branches=False)
                batch_loss += loss_c
                
            total_loss += batch_loss / 10.0
            batches += 1
            
        print(f"  Epoch {epoch+1}/5 | Local Error: {total_loss/batches:.4f}")
        
    # Evaluate
    correct = 0
    total = 0
    for data, target in test_loader:
        data = data.view(data.size(0), -1)
        outs = []
        for m in models:
            outs.append(m.forward(data))
        out = torch.cat(outs, dim=1)
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
    print(f"---> DLL Offline Accuracy: {100.0 * correct / total:.2f}%\n")

# ==============================================================================
# 2. BACKPROP OFFLINE TRAINING
# ==============================================================================
def run_backprop_offline():
    print("\n" + "="*50)
    print("2. BACKPROP OFFLINE TRAINING (Baseline)")
    print("="*50)
    train_loader, test_loader = get_full_mnist()
    
    model = StandardMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        total_loss = 0.0
        for data, target in train_loader:
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f}")
        
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"---> Backprop Offline Accuracy: {100.0 * correct / total:.2f}%\n")

# ==============================================================================
# 3. BACKPROP CLASS-INCREMENTAL LEARNING (CIL)
# ==============================================================================
def run_backprop_cil():
    print("\n" + "="*50)
    print("3. BACKPROP CLASS-INCREMENTAL LEARNING (Catastrophic Forgetting)")
    print("="*50)
    tasks = get_split_mnist_tasks()
    
    model = StandardMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    cil_accuracies = []
    
    for task_idx, task in enumerate(tasks):
        print(f"\n--- Phase: Task {task_idx+1} (Classes {task['classes']}) ---")
        for epoch in range(5):
            total_loss = 0.0
            for data, target in task['train_loader']:
                data = data.view(data.size(0), -1)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(task['train_loader']):.4f}")
            
        # Evaluate CIL Accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for t in range(task_idx + 1):
                for data, target in tasks[t]['test_loader']:
                    data = data.view(data.size(0), -1)
                    out = model(data)
                    pred = out.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    
        acc = 100.0 * correct / total
        cil_accuracies.append(acc)
        print(f"---> Backprop CIL Accuracy after Task {task_idx+1}: {acc:.2f}%")
        
    print("\nFINAL BACKPROP CIL PROFILE:")
    for i, acc in enumerate(cil_accuracies):
        print(f"  After Task {i+1}: {acc:.2f}%")

if __name__ == "__main__":
    run_dll_offline_paper()
    # run_evodll_offline() # Skip the imbalanced one to save time
    # run_backprop_offline() # Already ran this, it gets 97%
    # run_backprop_cil() # Already ran this, it gets 19%
