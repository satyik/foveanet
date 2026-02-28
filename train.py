import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from foveanet import FoveaNetDelta
from foveanet.data import get_dataloaders

def get_args():
    parser = argparse.ArgumentParser(description="Train FoveaNet-Ψ natively on CIFAR-100")
    parser.add_argument('--data-path', type=str, default='', help='Path to dataset. If empty, auto-downloads CIFAR-100.')
    parser.add_argument('--subset-size', type=int, default=None, help='Number of samples to run (for a quick subset test).')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='Total number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # In FoveaNet-Delta, the multi-step Phase 3 T=2 loop is handled natively inside 
        # the model's forward pass. In training, it returns auxiliary prediction targets.
        outputs, expected_nodes, actual_nodes = model(images)
        
        # 1. Standard Negative Log-Likelihood (Classification Loss)
        nll_loss = nn.NLLLoss()(outputs, targets)
        
        # 2. Predictive Coding Prior Reconstruction Loss
        # Instructs the Generative Prior to predict the unmasked graph node mean
        prior_loss = nn.MSELoss()(expected_nodes, actual_nodes.detach())
        
        loss = nll_loss + 0.05 * prior_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(images)}/{len(dataloader.dataset)}] Loss: {loss.item():.6f} Acc: {100. * correct / total:.2f}%")
            
    return total_loss / len(dataloader), 100. * correct / total, time.time() - start_time


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = nn.NLLLoss()(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / len(dataloader), 100. * correct / total, time.time() - start_time


def main():
    args = get_args()
    args.num_classes = 100 # Hardcode CIFAR-100

    print(f"Initializing FoveaNet-Δ Model for Training with {args.num_classes} classes...")
    model = FoveaNetDelta(num_classes=args.num_classes)
    
    # Metal (MPS) on Apple Silicon, CUDA on Nvidia, CPU fallback
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Dataloaders
    train_loader, val_loader, train_len, val_len, using_autodownload = get_dataloaders(args)
    print(f"Training on {train_len} images, Validating on {val_len} images.")

    # Optimizer (Generation D specified parameters)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Epoch Loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # 3-Phase Curriculum Schedule Management (Generation D)
        if epoch <= 10:
            current_phase = 1
            print("Phase 1: Dense Routing ON | Prior Error Tracking OFF")
            # Freeze prior net early to stop it hallucinating garbage into the GIN layers
            for param in model.router.prior_net.parameters():
                param.requires_grad = False
        elif epoch <= 25:
            current_phase = 2
            print("Phase 2: Error-Routing (Top 50%) ON | Prior Unfrozen | Belief EMA ON")
            for param in model.router.prior_net.parameters():
                param.requires_grad = True
        else:
            current_phase = 3
            print("Phase 3: Dynamic Threshold (mu + 0.5sigma) Error-Routing ON | T=2 Active")
            
        model.set_training_phase(current_phase)
        
        train_loss, train_acc, train_time = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train Complete: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Time={train_time:.1f}s")
        
        val_loss, val_acc, val_time = evaluate(model, val_loader, device)
        print(f"Validation Complete: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Time={val_time:.1f}s")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New Best Validation Accuracy: {best_val_acc:.2f}%!")
            
    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
