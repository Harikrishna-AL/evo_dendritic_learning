import torch
from dendritic_neuron import EvoDLLNeuron

def test_evo_dll():
    # 1. Initialize neuron
    neuron = EvoDLLNeuron(input_dim=2, initial_branches=1, lr=0.05)
    
    # 2. Simple Dataset (Pattern A: XOR-like subset)
    # [0, 0] -> 0
    # [1, 1] -> 1
    X_A = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y_A = torch.tensor([[0.0], [1.0]])
    
    print("--- Training Pattern A ---")
    for epoch in range(300):
        out = neuron.forward(X_A)
        loss = neuron.backward_and_update(y_A, freeze_old_branches=False)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Local Error = {loss:.4f}, Output = {out.detach().numpy().flatten()}")
            
    print(f"\nFinal Branch Count: {neuron.num_branches()}")
    
    # 3. New Dataset (Pattern B: requires new structure)
    # [0, 1] -> 1
    # [1, 0] -> 0
    X_B = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    y_B = torch.tensor([[1.0], [0.0]])
    
    print("\n--- Training Pattern B (with autonomous growth) ---")
    for epoch in range(300):
        out = neuron.forward(X_B)
        loss = neuron.backward_and_update(y_B, freeze_old_branches=True)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Local Error = {loss:.4f}, Branches = {neuron.num_branches()}, Output = {out.detach().numpy().flatten()}")

    print("\n--- Testing Backward Transfer (Checking Forgetting on Pattern A) ---")
    out_A_after = neuron.forward(X_A)
    # The targets for Pattern A were [[0.0], [1.0]]
    error_A_after = torch.mean((y_A - out_A_after) ** 2).item()
    print(f"Pattern A Error After Pattern B: {error_A_after:.4f}")
    print(f"Pattern A Output After Pattern B: {out_A_after.detach().numpy().flatten()}")
    print("If the error remains low and outputs are close to [0.0, 1.0], zero-forgetting is achieved!")

if __name__ == "__main__":
    test_evo_dll()
