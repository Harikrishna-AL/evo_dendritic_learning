import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from evolved_coordinator import EvolvedCoordinator
from evo_dll_network import EvoDLLNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
        train_subset = Subset(train_dataset, train_indices)
        
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
        test_subset = Subset(test_dataset, test_indices)
        
        tasks.append({
            'classes': classes,
            'train_loader': DataLoader(train_subset, batch_size=256, shuffle=True),
            'test_loader': DataLoader(test_subset, batch_size=1000, shuffle=False)
        })
        
    return tasks


def evaluate_evolved_cil(models, coordinator, tasks, num_tasks_seen):
    """
    Evaluates CIL accuracy using the TRINITY-style Evolved Coordinator.
    """
    correct = 0
    total = 0
    
    for t in range(num_tasks_seen):
        test_loader = tasks[t]['test_loader']
        for data, target in test_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            target = target.to(device)
            
            # 1. The Coordinator routes the image to a Task ID
            predicted_task_ids = coordinator.route(data)
            
            # Since batches might be routed to different tasks, we process sample by sample
            for i in range(data.size(0)):
                x_i = data[i:i+1]
                t_pred = predicted_task_ids[i].item()
                
                # Prevent out-of-bounds routing if coordinator predicts a future task
                if t_pred >= num_tasks_seen:
                    t_pred = num_tasks_seen - 1
                
                # 2. Wake up the specific Frozen Experts for that Task
                expert_classes = tasks[t_pred]['classes']
                c1, c2 = expert_classes[0], expert_classes[1]
                
                out_c1 = models[c1].forward(x_i)
                out_c2 = models[c2].forward(x_i)
                
                # 3. Take the argmax between the two experts
                if out_c1.item() > out_c2.item():
                    final_pred = c1
                else:
                    final_pred = c2
                    
                if final_pred == target[i].item():
                    correct += 1
                total += 1
                
    return 100.0 * correct / total

def train_split_mnist_evolved():
    print("Preparing Split-MNIST tasks...")
    tasks = get_split_mnist_tasks()
    
    # 1. Initialize 10 physically independent, isolated Expert Networks
    models = [EvoDLLNetwork(input_dim=784, output_dim=1, initial_branches=64, lr=0.01, device=device) for _ in range(10)]
    
    # 2. Initialize the TRINITY Evolved Coordinator
    coordinator = EvolvedCoordinator(input_dim=784, num_tasks=5, pop_size=100, sigma=0.1, lr=0.5, device=device)
    
    for task_id, task in enumerate(tasks):
        active_classes = task['classes']
        print(f"\n==========================================================")
        print(f" Phase: Task {task_id+1} (Classes {active_classes})")
        print(f"==========================================================")
        
        train_loader = task['train_loader']
        
        # --- Train the Frozen Experts (DLL Localized Learning) ---
        print("  [EXPERTS] Training specialized DLL experts...")
        for epoch in range(10):
            total_loss = 0.0
            batches = 0
            for data, target in train_loader:
                data = data.to(device)
                data = data.view(data.size(0), -1)
                
                # Create true one-hot [Batch, 10]
                target_onehot = torch.zeros(target.size(0), 10).to(device)
                target_onehot.scatter_(1, target.to(device).unsqueeze(1), 1.0)
                
                # Train only the two active experts
                for c in active_classes:
                    target_onehot_c = target_onehot[:, c].unsqueeze(1)
                    models[c].forward(data)
                    loss_c = models[c].backward_and_update(target_onehot_c, active_classes=[0])
                    total_loss += loss_c
                batches += 1
                
        print("  [EXPERTS] Freezing experts structurally.")
        
        # --- Evolve the Coordinator (Evolutionary Strategy) ---
        print("  [COORDINATOR] Evolving router to map images to Task ID...")
        # Get a full batch of current task data for evolution
        all_data = []
        for data, _ in train_loader:
            data = data.to(device).view(data.size(0), -1)
            all_data.append(data)
        current_task_data = torch.cat(all_data, dim=0)
        
        # Evolve the router weights using a fast 200-image subset
        subset_idx = torch.randperm(current_task_data.size(0))[:200]
        evolution_subset = current_task_data[subset_idx]
        
        coordinator.evolve(evolution_subset, current_task_id=task_id, iterations=50)
        
        # Save a microscopic buffer (20 images) to prevent the router itself from forgetting
        coordinator.add_to_buffer(current_task_data, task_id=task_id, num_samples=20)
        
        # Evaluate
        cil_acc = evaluate_evolved_cil(models, coordinator, tasks, num_tasks_seen=task_id+1)
        print(f"\n---> Class-Incremental Accuracy after Task {task_id+1}: {cil_acc:.2f}%\n")
        
if __name__ == '__main__':
    train_split_mnist_evolved()
