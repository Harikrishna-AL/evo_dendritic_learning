import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from dendritic_layer import DendriticLayer

# Set up simple device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_split_mnist_tasks():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    tasks = []
    # Split into 5 tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
    for t in range(5):
        classes = [2*t, 2*t + 1]
        
        # Filter train
        train_indices = [i for i, label in enumerate(train_dataset.targets) if label in classes]
        train_subset = Subset(train_dataset, train_indices)
        
        # Filter test
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
        test_subset = Subset(test_dataset, test_indices)
        
        tasks.append({
            'classes': classes,
            'train_loader': DataLoader(train_subset, batch_size=64, shuffle=True),
            'test_loader': DataLoader(test_subset, batch_size=1000, shuffle=False)
        })
        
    return tasks

class DendriticNetwork(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=50):
        super().__init__()
        # The hidden layer is where our dendritic structural plasticity lives
        self.hidden = DendriticLayer(in_dim, hidden_dim, initial_branches=1)
        
        # Multi-head output (Class-Incremental: we will add neurons here for new classes)
        # We start with 0 output classes and add them dynamically
        self.classifier = nn.Linear(hidden_dim, 0)
        self.hidden_dim = hidden_dim
        
    def add_classes(self, num_new_classes):
        """Grow the output layer for new classes."""
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        
        out_features = self.classifier.out_features + num_new_classes
        new_classifier = nn.Linear(self.hidden_dim, out_features).to(device)
        
        # Copy old weights
        if self.classifier.out_features > 0:
            new_classifier.weight.data[:self.classifier.out_features] = old_weight
            new_classifier.bias.data[:self.classifier.out_features] = old_bias
            
        self.classifier = new_classifier
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten image
        h = self.hidden(x)
        return self.classifier(h)

def train_continuous(model, train_loader, task_classes, epochs=3):
    """Fast Phase: Train continuous weights and the output head."""
    model.train()
    # We only train the continuous parameters that are unfrozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # CRITICAL FIX: Prevent the classifier from forgetting old classes!
            # CrossEntropyLoss will try to push logits of old classes to negative infinity.
            # We must zero out the gradients for the old classes in the output layer.
            if model.classifier.weight.grad is not None:
                mask = torch.zeros_like(model.classifier.weight.grad)
                mask[task_classes] = 1.0
                model.classifier.weight.grad *= mask
                model.classifier.bias.grad *= mask[:, 0] if mask.dim() > 1 else mask
                
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

def evaluate(model, test_loader, task_classes=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if task_classes is not None:
                # Task-Incremental Learning (TIL) evaluation
                # We only consider the logits for the current task's classes
                task_output = output[:, task_classes]
                # Map the argmax back to the original class indices
                pred = torch.tensor(task_classes, device=device)[task_output.argmax(dim=1, keepdim=True)]
            else:
                pred = output.argmax(dim=1, keepdim=True)
                
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

if __name__ == "__main__":
    print(f"Using device: {device}")
    tasks = get_split_mnist_tasks()
    model = DendriticNetwork(hidden_dim=20).to(device)
    
    # Task 1: Digits 0 and 1
    task_idx = 0
    task = tasks[task_idx]
    print(f"\n--- Phase 1: Task {task_idx+1} (Classes {task['classes']}) ---")
    
    # Grow output head for the 2 new classes
    model.add_classes(2)
    
    train_continuous(model, task['train_loader'], task['classes'], epochs=3)
    acc = evaluate(model, task['test_loader'], task['classes'])
    print(f"Task {task_idx+1} Test Accuracy: {acc:.2f}%")
    
    print("\nFreezing continuous weights of the hidden dendritic layer...")
    for param in model.hidden.parameters():
        param.requires_grad = False
        
    # Task 2: Digits 2 and 3
    task_idx = 1
    task = tasks[task_idx]
    print(f"\n--- Phase 2: Task {task_idx+1} (Classes {task['classes']}) ---")
    
    model.add_classes(2)
    
    # To properly evaluate the EA routing and structural addition, 
    # we would apply it here. For the skeleton, let's see if the linear 
    # output head can solve Task 2 using the FROZEN hidden features of Task 1!
    print("Training ONLY the new output classification head on Task 2 (using frozen Task 1 features)...")
    train_continuous(model, task['train_loader'], task['classes'], epochs=3)
    
    acc2 = evaluate(model, task['test_loader'], task['classes'])
    print(f"Task 2 Test Accuracy (Zero structural growth): {acc2:.2f}%")
    
    if acc2 < 90.0:
        print("\nAccuracy is poor! This triggers the Evolutionary Structural Allocation!")
        print("Adding a branch to all hidden neurons...")
        model.hidden.add_branch_to_all()
        
        print("Unfreezing only the NEW structural components (mature branches stay frozen)...")
        for neuron in model.hidden.neurons:
            # CRITICAL FIX: Freeze soma_bias to preserve old task calibration
            neuron.soma_bias.requires_grad = False
            
            # Initialize new branch to overcome the frozen hyperpolarized soma_bias
            with torch.no_grad():
                neuron.branches[-1].branch_weight.data.fill_(
                    abs(neuron.soma_bias.item()) + 1.0
                )
                
            for param in neuron.branches[-1].parameters():
                param.requires_grad = True
            
        print("Retraining with new structural capacity...")
        train_continuous(model, task['train_loader'], task['classes'], epochs=3)
        acc2_grown = evaluate(model, task['test_loader'], task['classes'])
        print(f"Task 2 Test Accuracy (After Structural Growth): {acc2_grown:.2f}%")
        
        # Check forgetting on Task 1
        acc1_after = evaluate(model, tasks[0]['test_loader'], tasks[0]['classes'])
        print(f"Task 1 Test Accuracy (Backward Transfer check): {acc1_after:.2f}%")
