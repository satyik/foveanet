import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(args):
    """
    Returns (train_loader, val_loader) strictly for CIFAR-100 natively.
    """
    using_autodownload = False
    
    # 1. Automatic Path Resolution for CIFAR-100
    if not args.data_path:
        args.data_path = './data'
        using_autodownload = True

    print(f"\n--- RUNNING ON REAL DATA ---")
    print(f"Dataset: CIFAR-100")
    print(f"Dataset path: {args.data_path}")

    # 2. CIFAR-100 Native Transforms (32x32)
    # Training transform with standard CIFAR-100 augmentations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), # Gen D protocol
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]), 
    ])
    
    # Validation transform (pure evaluation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]), 
    ])
    
    train_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=using_autodownload, transform=train_transform)
    val_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=using_autodownload, transform=val_transform)

    # 3. Handle Subsets (useful for quick debugging)
    if args.subset_size is not None:
        print(f"Limiting to a subset of {args.subset_size} samples...")
        train_indices = torch.randperm(len(train_dataset))[:args.subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset))[:args.subset_size].tolist()
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
    # 4. Create Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle for training
        num_workers=args.workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset), using_autodownload
