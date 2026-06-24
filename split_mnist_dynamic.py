import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from dynamic_unified_dll import DynamicUnifiedDLL

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
        
        train_indices = [i for i, label in enumerate(train_dataset.targets) if label in classes]
        train_subset = Subset(train_dataset, train_indices)
        
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
        test_subset = Subset(test_dataset, test_indices)
        
        tasks.append({
            'classes': classes,
            'train_loader': DataLoader(train_subset, batch_size=256, shuffle=True),
            'test_loader': DataLoader(test_subset, batch_size=1000, shuffle=False)
        })
        
    return tasks

def evaluate_cil(model, tasks, num_tasks_seen):
    """
    Evaluates Class-Incremental Learning accuracy.
    No Task ID is given during inference. We feed the image to the unified network.
    The single Softmax output handles the 'Open Set Problem' naturally.
    """
    correct = 0
    total = 0
    
    # For measuring interference
    print("\n  [INTERFERENCE LOG] Average Logits per Task during Evaluation:")
    
    for t in range(num_tasks_seen):
        test_loader = tasks[t]['test_loader']
        
        task_logits_sum = torch.zeros(10).to(device)
        task_samples = 0
        
        for data, target in test_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            
            # Forward pass through the UNIFIED deep network
            out = model.forward(data) # [Batch, 10]
            
            # Prediction is simply the argmax over ALL 10 outputs
            pred = out.argmax(dim=1)
            
            correct += (pred == target.to(device)).sum().item()
            total += target.size(0)
            
            task_logits_sum += out.sum(dim=0)
            task_samples += target.size(0)
            
        avg_logits = task_logits_sum / task_samples
        print(f"    Evaluating on Task {t+1} (Classes {tasks[t]['classes']}):")
        
        # Print the average activation for each task's classes
        logits_str = []
        for past_t in range(num_tasks_seen):
            classes = tasks[past_t]['classes']
            val = (avg_logits[classes[0]] + avg_logits[classes[1]]) / 2.0
            logits_str.append(f"T{past_t+1} logits: {val:.4f}")
            
        print(f"      -> {', '.join(logits_str)}")
            
    return 100.0 * correct / total

def train_split_mnist_dynamic():
    print("Loading Split-MNIST...")
    tasks = get_split_mnist_tasks()
    
    # Initialize 1 Unified Multi-Layer Network
    print("Initializing Dynamic Unified DLL Network (784 -> 64 -> 64 -> 10)...")
    model = DynamicUnifiedDLL(input_dim=784, output_dim=10, initial_hidden=64, lr=0.005, device=device)
    
    # We will track CIL accuracy after each task
    cil_accuracies = []
    
    for task_idx, task in enumerate(tasks):
        active_classes = task['classes']
        print(f"\n==========================================================")
        print(f" Phase: Task {task_idx+1} (Classes {active_classes})")
        print(f"==========================================================")
        
        # If this is a new task, FREEZE old features and EXPAND network capacity!
        if task_idx > 0:
            print(f"  [NEUROGENESIS] Freezing old {model.current_H} neurons.")
            model.freeze_current_state(tasks[task_idx - 1]['classes'])
            
            new_neurons = 32
            print(f"  [NEUROGENESIS] Sprouting {new_neurons} new neurons in hidden layers!")
            model.expand(new_neurons)
            print(f"  [NEUROGENESIS] New Hidden Size: {model.current_H}")
        
        # We'll do 20 epochs per task
        for epoch in range(20):
            total_loss = 0.0
            batches = 0
            
            for data, target in task['train_loader']:
                data = data.to(device)
                data = data.view(data.size(0), -1)
                
                # Create one-hot targets for the unified cross-entropy output
                target_onehot = torch.zeros(target.size(0), 10).to(device)
                target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
                
                # Single pass update
                loss = model.backward_and_update(data, target_onehot, active_classes)
                
                total_loss += loss
                batches += 1
                
            avg_loss = total_loss / batches
            print(f"  Epoch {epoch+1}/20 | Cross-Entropy Loss: {avg_loss:.4f} | Hidden Neurons: {model.current_H}")
            
        # Evaluate CIL Accuracy
        acc = evaluate_cil(model, tasks, num_tasks_seen=task_idx+1)
        cil_accuracies.append(acc)
        print(f"\n---> Class-Incremental Accuracy after Task {task_idx+1}: {acc:.2f}%")
        
    print("\n==========================================================")
    print(" FINAL CIL ACCURACY PROFILE (Unified Progressive DLL):")
    for i, acc in enumerate(cil_accuracies):
        print(f"  After Task {i+1}: {acc:.2f}%")
        
if __name__ == "__main__":
    train_split_mnist_dynamic()
