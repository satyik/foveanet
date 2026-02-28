import time
import argparse
import torch

from foveanet import FoveaNetOmega
from foveanet.data import get_dataloaders

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate FoveaNet-Ω natively on CIFAR-100")
    parser.add_argument('--data-path', type=str, default='', help='Path to dataset. If empty, will auto-download CIFAR-100.')
    parser.add_argument('--subset-size', type=int, default=None, help='Number of samples to run (for a subset). If None, run full dataset (10,000 val samples).')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    return parser.parse_args()


def main():
    args = get_args()
    
    # Hardcode CIFAR-100
    args.num_classes = 100

    print(f"Initializing FoveaNet-Ω Model with {args.num_classes} classes...")
    model = FoveaNetOmega(num_classes=args.num_classes)
    
    # Automatically switch to Metal (MPS) on Apple Silicon, CUDA on Nvidia, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()

    dataloader, dataset_len, using_autodownload = get_dataloaders(args)

    print(f"Evaluating on {dataset_len} images...")
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"Batch {i + 1}/{len(dataloader)} - Accuracy so far: {100. * correct / total:.2f}%")

    end_time = time.time()
    print("-" * 50)
    if using_autodownload:
        print("Note: Accuracy will be low because the model is initialized with random weights and hasn't been trained!")
    print(f"Final Validation Accuracy: {100. * correct / total:.2f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Throughput: {dataset_len / (end_time - start_time):.2f} images/sec")

if __name__ == "__main__":
    main()
