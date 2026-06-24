import torch

class EvolvedCoordinator:
    def __init__(self, input_dim=784, num_tasks=5, pop_size=50, sigma=0.1, lr=0.5, device="cpu"):
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.device = device
        
        # The central routing weights [784, 5]
        self.weights = torch.zeros(input_dim, num_tasks).to(device)
        
        # Micro-fitness buffer: lists of (x, task_id)
        self.fitness_buffer_x = []
        self.fitness_buffer_y = []
        
    def add_to_buffer(self, x, task_id, num_samples=20):
        """Adds a microscopic handful of prototypes for a task to the fitness buffer."""
        # x is [Batch, 784]
        # We only take `num_samples` to keep the buffer mathematically trivial
        idx = torch.randperm(x.size(0))[:min(num_samples, x.size(0))]
        self.fitness_buffer_x.append(x[idx].clone().to(self.device))
        
        y = torch.full((x[idx].size(0),), task_id, dtype=torch.long, device=self.device)
        self.fitness_buffer_y.append(y)
        
    def evolve(self, current_x, current_task_id, iterations=50):
        """
        Runs the Evolutionary Strategy to update the routing weights.
        Optimizes for accuracy on the current task + the micro-fitness buffer.
        """
        # 1. Gather all fitness evaluation data
        all_x = [current_x] + self.fitness_buffer_x
        all_y = [torch.full((current_x.size(0),), current_task_id, dtype=torch.long, device=self.device)] + self.fitness_buffer_y
        
        eval_x = torch.cat(all_x, dim=0)
        eval_y = torch.cat(all_y, dim=0)
        
        best_reward = 0.0
        
        # 2. ES Loop (Vectorized for extreme performance)
        for epoch in range(iterations):
            # Generate Population of perturbed weights
            # noise shape: [pop_size, 784, 5]
            noise = torch.randn(self.pop_size, self.input_dim, self.num_tasks, device=self.device)
            w_candidates = self.weights.unsqueeze(0) + self.sigma * noise
            
            # Evaluate entire population in a single batched matrix multiplication
            # eval_x: [Batch, 784]
            # w_candidates: [pop_size, 784, 5]
            # logits: [pop_size, Batch, 5]
            logits = torch.einsum('bi,pio->pbo', eval_x, w_candidates)
            preds = logits.argmax(dim=2)
            
            # Fitness is routing accuracy
            # eval_y: [Batch] -> [1, Batch]
            accuracy = (preds == eval_y.unsqueeze(0)).float().mean(dim=1)
            rewards = accuracy
            
            best_reward = rewards.max().item()
            
            # Standardize rewards for stable ES updates
            if rewards.std() > 1e-6:
                norm_rewards = (rewards - rewards.mean()) / rewards.std()
            else:
                norm_rewards = rewards - rewards.mean()
                
            # Update weights using Natural Evolution Strategies formula
            weighted_noise = noise * norm_rewards.view(-1, 1, 1)
            grad_estimate = weighted_noise.mean(dim=0) / self.sigma
            
            self.weights += self.lr * grad_estimate
            
        print(f"    [Coordinator] Evolution complete. Best Routing Accuracy: {best_reward*100:.2f}%")
            
    def route(self, x):
        """Returns the assigned Task ID for an input batch."""
        logits = torch.matmul(x, self.weights)
        return logits.argmax(dim=1)
