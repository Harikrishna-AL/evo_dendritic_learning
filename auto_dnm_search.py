import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from auto_dnm_layer import AutoDNMLayer

class AutoDNMSearch:
    """
    A Memetic Evolutionary Algorithm (NAS + Backprop) to discover the optimal 
    Dendritic Neuron Architecture and parameters simultaneously.
    """
    def __init__(self, input_dim, num_branches=5, pop_size=20, device="cpu"):
        self.input_dim = input_dim
        self.num_branches = num_branches
        self.pop_size = pop_size
        self.device = device
        
        self.population = []
        for _ in range(pop_size):
            self.population.append(self._create_random_individual())
            
    def _create_random_individual(self):
        # 3 choices for Synapse, 3 for Dendrite, 3 for Membrane, 3 for Soma
        return AutoDNMLayer(
            input_dim=self.input_dim,
            num_branches=self.num_branches,
            syn_func=random.randint(0, 2),
            den_func=random.randint(0, 2),
            mem_func=random.randint(0, 2),
            soma_func=random.randint(0, 2)
        ).to(self.device)
        
    def _mutate_individual(self, parent):
        child = copy.deepcopy(parent)
        
        # 1. Structural Mutation (Architecture DNA)
        mutated_structure = False
        if random.random() < 0.3:
            # Mutate one random layer
            layer_to_mutate = random.randint(0, 3)
            if layer_to_mutate == 0:
                child.syn_func = random.randint(0, 2)
            elif layer_to_mutate == 1:
                child.den_func = random.randint(0, 2)
            elif layer_to_mutate == 2:
                child.mem_func = random.randint(0, 2)
            else:
                child.soma_func = random.randint(0, 2)
            mutated_structure = True
            
        # If structure changed, it's often best to reset the weights since 
        # the mathematical landscape has entirely shifted
        if mutated_structure:
            child.w.data = torch.randn_like(child.w.data) * 0.1
            child.theta.data = torch.randn_like(child.theta.data) * 0.1
        else:
            # 2. Weight Mutation (Parameter DNA)
            if random.random() < 0.5:
                child.w.data += torch.randn_like(child.w.data) * 0.05
                child.theta.data += torch.randn_like(child.theta.data) * 0.05
                
        return child

    def evaluate_fitness(self, individual, X, y, epochs=5):
        """
        Memetic evaluation: We use Backprop (Adam) to briefly train the 
        individual's weights before assessing its final fitness.
        """
        optimizer = optim.Adam(individual.parameters(), lr=0.05)
        criterion = nn.BCELoss() if individual.soma_func == 0 else nn.MSELoss()
        
        individual.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = individual(X)
            
            # If step function or tanh is used, BCE will fail. 
            # We map target to -1, 1 for Tanh, or use MSE.
            if individual.soma_func == 0:
                loss = criterion(out, y)
            else:
                # For hard step or Tanh, use MSE to pull logits toward target
                loss = criterion(out, y)
                
            loss.backward()
            optimizer.step()
            
        # Calculate final accuracy as fitness
        individual.eval()
        with torch.no_grad():
            out = individual(X)
            if individual.soma_func == 0:
                preds = (out > 0.5).float()
            elif individual.soma_func == 1:
                preds = out # already 0 or 1
            else:
                preds = (out > 0.0).float()
                
            accuracy = (preds == y).float().mean().item()
            
        return accuracy

    def evolve(self, X, y, generations=20):
        print(f"Starting AutoDNM Search for {generations} generations...")
        
        for gen in range(generations):
            fitness_scores = []
            
            for i, ind in enumerate(self.population):
                fit = self.evaluate_fitness(ind, X, y, epochs=5)
                fitness_scores.append((fit, ind))
                
            # Sort descending by fitness
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            best_fit = fitness_scores[0][0]
            best_arch = fitness_scores[0][1].get_architecture_string()
            
            print(f"Gen {gen+1}/{generations} | Best Accuracy: {best_fit*100:.2f}% | Arch: {best_arch}")
            
            # Elitism: Keep top 20%
            elite_count = int(self.pop_size * 0.2)
            new_population = [item[1] for item in fitness_scores[:elite_count]]
            
            # Generate the rest
            while len(new_population) < self.pop_size:
                # Select a random elite parent
                parent = random.choice(fitness_scores[:elite_count])[1]
                child = self._mutate_individual(parent)
                new_population.append(child)
                
            self.population = new_population
            
        print("Evolution Complete!")
        # Return the absolute best model
        return fitness_scores[0][1], fitness_scores[0][0]
