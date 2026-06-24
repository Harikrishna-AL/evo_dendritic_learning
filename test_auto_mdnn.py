import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auto_mdnn_search import AutoMDNNSearch

def main():
    print("Loading Digits Dataset...")
    # Digits dataset: 8x8 images of digits 0-9
    data = load_digits()
    X = data.data
    y = data.target
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) # CrossEntropyLoss expects long
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    input_dim = X.shape[1]
    num_classes = 10
    
    print(f"Dataset ready. Inputs: {input_dim}, Classes: {num_classes}, Train samples: {len(X_train)}")
    
    # Initialize the NAS Evolutionary Search
    # 20 brains, 10 branches each
    searcher = AutoMDNNSearch(
        input_dim=input_dim, 
        num_classes=num_classes, 
        num_branches=10, 
        pop_size=20
    )
    
    # Run the evolution for 15 generations
    best_model, best_acc = searcher.evolve(X_train, y_train, generations=15)
    
    print("\n" + "="*50)
    print("EVOLUTION COMPLETE: WINNING MULTI-CLASS NEURON NETWORK DISCOVERED")
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
