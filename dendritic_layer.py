import torch
import torch.nn as nn
from dendritic_neuron import DendriticNeuron

class DendriticLayer(nn.Module):
    r"""
    A full layer of Dendritic Neurons.
    Acts as a drop-in replacement for nn.Linear, but with structural plasticity.
    """
    def __init__(self, in_features, out_features, initial_branches=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # A layer is simply a collection of independent Dendritic Neurons
        self.neurons = nn.ModuleList([
            DendriticNeuron(in_features, initial_branches) for _ in range(out_features)
        ])
        
    def add_output_neuron(self, initial_branches=1):
        """
        For Class-Incremental Learning: Add a brand new output neuron 
        when a new class is encountered.
        """
        new_neuron = DendriticNeuron(self.in_features, initial_branches)
        if len(self.neurons) > 0:
            device = next(self.neurons[0].parameters()).device
            new_neuron.to(device)
            
        self.neurons.append(new_neuron)
        self.out_features += 1
        print(f"Layer grew a new output neuron! Total outputs: {self.out_features}")
        
    def add_branch_to_all(self):
        """
        Expands the structural capacity of all neurons in this layer.
        """
        for neuron in self.neurons:
            neuron.add_branch()
            
    def forward(self, x):
        """
        Computes the forward pass for the entire layer.
        x shape: (batch_size, in_features)
        output shape: (batch_size, out_features)
        """
        # Collect outputs from each independent neuron
        outputs = [neuron(x) for neuron in self.neurons]
        
        # Concatenate along the feature dimension
        return torch.cat(outputs, dim=1)
