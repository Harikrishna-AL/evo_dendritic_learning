import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auto_mdnn_search import AutoMDNNSearch

def make_checkerboard(n_samples=2000, noise=0.1):
    """
    Creates a 2D checkerboard pattern. 
    This is mathematically a highly periodic dataset.
    """
    X = np.random.uniform(-4, 4, size=(n_samples, 2))
    # Checkerboard logic: if sin(x)*sin(y) > 0, class 1, else class 0
    y = (np.sin(X[:, 0] * 1.5) * np.sin(X[:, 1] * 1.5) > 0).astype(int)
    X += np.random.normal(0, noise, size=X.shape)
    return X, y

def main():
    print("Loading Periodic Checkerboard Dataset...")
    X, y = make_checkerboard(n_samples=2000, noise=0.1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    input_dim = 2
    num_classes = 2
    
    print(f"Dataset ready. Inputs: {input_dim}, Classes: {num_classes}, Train samples: {len(X_train)}")
    
    # Initialize the NAS Evolutionary Search
    # 20 brains, 5 branches each
    searcher = AutoMDNNSearch(
        input_dim=input_dim, 
        num_classes=num_classes, 
        num_branches=5, 
        pop_size=20
    )
    
    # Run the evolution for 15 generations
    best_model, best_acc = searcher.evolve(X_train, y_train, generations=15)
    
    print("\n" + "="*50)
    print("EVOLUTION COMPLETE: CHECKERBOARD NEURON DISCOVERED")
    print("="*50)
    print(f"Architecture: {best_model.get_architecture_string()}")
    print(f"Training Accuracy: {best_acc*100:.2f}%")
    
    # Evaluate on test set
    best_model.eval()
    with torch.no_grad():
        logits = best_model(X_test)
        preds = logits.argmax(dim=1)
        test_acc = (preds == y_test).float().mean().item()
        
    print(f"Test Set Accuracy: {test_acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
