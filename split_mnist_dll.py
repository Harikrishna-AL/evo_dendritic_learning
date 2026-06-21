import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from evo_dll_network import EvoDLLNetwork

device = "cpu"

def get_split_mnist_tasks():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
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
            'train_loader': DataLoader(train_subset, batch_size=256, shuffle=True),
            'test_loader': DataLoader(test_subset, batch_size=1000, shuffle=False)
        })
        
    return tasks

def evaluate_cil(models, tasks, num_tasks_seen):
    """
    Evaluates Class-Incremental Learning accuracy.
    No Task ID is given during inference. We feed the image to all 10 independent networks.
    Each network outputs a scalar (its confidence). We take the argmax.
    """
    correct = 0
    total = 0
    
    for t in range(num_tasks_seen):
        test_loader = tasks[t]['test_loader']
        
        for data, target in test_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            # Forward pass through all 10 independent networks
            outs = []
            for m in models:
                outs.append(m.forward(data))
            
            # outs is a list of [Batch, 1] tensors. Concatenate to [Batch, 10]
            out = torch.cat(outs, dim=1)
            
            # Prediction is simply the argmax over ALL 10 outputs (True CIL)
            pred = out.argmax(dim=1)
            
            correct += (pred == target).sum().item()
            total += target.size(0)
            
    return 100.0 * correct / total

def train_split_mnist():
    print("Loading Split-MNIST...")
    tasks = get_split_mnist_tasks()
    
    # Initialize 10 independent Networks (True Biological Isolation)
    models = [EvoDLLNetwork(input_dim=784, output_dim=1, initial_branches=1, lr=0.01) for _ in range(10)]
    
    # We will track CIL accuracy after each task
    cil_accuracies = []
    
    for task_idx, task in enumerate(tasks):
        active_classes = task['classes']
        print(f"\n==========================================================")
        print(f" Phase: Task {task_idx+1} (Classes {active_classes})")
        print(f"==========================================================")
        
        # Only freeze if we have seen a previous task
        freeze = (task_idx > 0)
        if freeze:
            print("Freezing old branches for all active models!")
        
        # We'll do 20 epochs per task
        for epoch in range(20):
            total_loss = 0.0
            batches = 0
            
            for data, target in task['train_loader']:
                data = data.to(device)
                data = data.view(data.size(0), -1)
                
                # Train only the networks responsible for the active classes
                # For each active class, it should output 1.0 for its own images, and 0.0 for the OTHER active class.
                batch_loss = 0.0
                for c in active_classes:
                    model_c = models[c]
                    # Target is 1.0 if the image belongs to class c, else 0.0
                    y_c = (target == c).float().unsqueeze(1).to(device)
                    
                    out_c = model_c.forward(data)
                    # No active_classes mask needed, because output_dim is 1
                    loss_c = model_c.backward_and_update(y_c, active_classes=[0], freeze_old_branches=freeze)
                    batch_loss += loss_c
                    
                total_loss += batch_loss / len(active_classes)
                batches += 1
                
            avg_loss = total_loss / batches
            branches_str = ", ".join([f"M{c}:{models[c].num_branches()}" for c in active_classes])
            print(f"  Epoch {epoch+1}/20 | Local Error: {avg_loss:.4f} | Branches: {branches_str}")
            
        # Evaluate CIL Accuracy
        acc = evaluate_cil(models, tasks, num_tasks_seen=task_idx+1)
        cil_accuracies.append(acc)
        print(f"\n---> Class-Incremental Accuracy after Task {task_idx+1}: {acc:.2f}%")
        
    print("\n==========================================================")
    print(" FINAL CIL ACCURACY PROFILE:")
    for i, acc in enumerate(cil_accuracies):
        print(f"  After Task {i+1}: {acc:.2f}%")
        
if __name__ == "__main__":
    train_split_mnist()
