import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dendritic_layer import DendriticLayer
import argparse
import time

class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(model, train_loader, test_loader, epochs=3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{epochs} complete.")
            
    # Evaluate
    model.eval()
    correct_train, total_train = 0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
            
    train_acc = correct_train / total_train
    test_acc = correct_test / total_test
    return train_acc, test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"Loading {args.dataset.upper()} Dataset...")
    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 28 * 28
    num_classes = 10

    # 1. Deep DDNN
    model_ddnn = nn.Sequential(
        DendriticLayer(in_features=input_dim, out_features=128, num_branches=5),
        DendriticLayer(in_features=128, out_features=num_classes, num_branches=5)
    )
    
    # 2. Standard MLP (Matched Params)
    # DDNN Params: (128*5*784*2) + (10*5*128*2) = 1,003,520 + 12,800 = 1,016,320
    # MLP Params: (784*H + H) + (H*10 + 10) = 796*H + 10 = 1016320 => H = 1276
    model_mlp = StandardMLP(input_dim, hidden_dim=1276, num_classes=num_classes)

    print(f"Deep DDNN Parameter Count: {count_parameters(model_ddnn):,}")
    print(f"Standard MLP Parameter Count: {count_parameters(model_mlp):,}")
    
    print("\nTraining Deep DDNN (10 epochs)...")
    start = time.time()
    ddnn_train, ddnn_test = train_and_evaluate(model_ddnn, train_loader, test_loader, epochs=10, device=device)
    print(f"DDNN Train Acc: {ddnn_train*100:.2f}% | Test Acc: {ddnn_test*100:.2f}% (Time: {time.time()-start:.1f}s)")
    
    print("\nTraining Standard MLP (10 epochs)...")
    start = time.time()
    mlp_train, mlp_test = train_and_evaluate(model_mlp, train_loader, test_loader, epochs=10, device=device)
    print(f"MLP Train Acc: {mlp_train*100:.2f}% | Test Acc: {mlp_test*100:.2f}% (Time: {time.time()-start:.1f}s)")

if __name__ == "__main__":
    main()
