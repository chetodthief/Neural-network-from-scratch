"""
train.py - Training Pipeline for Custom Lane Segmentation Neural Network
Includes:
    - Custom Layer CNN trained from scratch (no pre-trained weights)
    - Data Augmentation, Batching, DataLoader
    - Combined BCE + Dice Loss for high IoU convergence
    - Metric calculation: IoU and Dice Score
    - TensorBoard integration (Loss, IoU, learning rate, sample images)
    - Epochs >= 30 as specified
    - Checkpoint saving (best & final weights)
    - Automatic plot generation for loss convergence curve
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CustomLaneSegNet
from dataset import get_dataset_splits


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for Binary Segmentation:
    Dice = (2 * |X & Y| + smooth) / (|X| + |Y| + smooth)
    Loss = 1 - Dice
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss + DiceLoss:
    Provides stable gradient updates (BCE) and directly optimizes region overlap (Dice).
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Computes batch mean Intersection-over-Union (IoU)."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        # Avoid division by zero when both prediction and target are empty
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple:
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_iou += compute_iou(logits, masks) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)
    return epoch_loss, epoch_iou


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item() * images.size(0)
            running_iou += compute_iou(logits, masks) * images.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_iou = running_iou / len(loader.dataset)
    return val_loss, val_iou


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    train_ious: list,
    val_ious: list,
    save_path: str = 'loss_convergence.png'
):
    """Generates and saves a clean, presentation-ready loss & IoU convergence figure."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r--', label='Validation (Test) Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss Convergence', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Combined Loss (BCE + Dice)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11)

    # IoU subplot
    ax2.plot(epochs, train_ious, 'g-', label='Training IoU', linewidth=2)
    ax2.plot(epochs, val_ious, 'm--', label='Validation (Test) IoU', linewidth=2)
    ax2.axhline(y=0.6, color='gray', linestyle='-.', alpha=0.8, label='Evaluation Threshold (IoU=0.6)')
    ax2.set_title('Intersection over Union (IoU) Progression', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Mean IoU', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Loss convergence graph successfully saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Custom Lane Segmentation Neural Network from Scratch")
    parser.add_argument('--dataset-dir', type=str, default='lane_dataset/dataset_seg', help="Path to dataset_seg directory")
    parser.add_argument('--epochs', type=int, default=35, help="Number of training epochs (>= 30 required)")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--img-width', type=int, default=64, help="Model input image width (default 64 per lecture notes)")
    parser.add_argument('--img-height', type=int, default=36, help="Model input image height (default 36 per lecture notes)")
    parser.add_argument('--log-dir', type=str, default='runs/lane_experiment', help="TensorBoard log directory")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help="Checkpoint output directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Enforce minimum 30 epochs requirement
    if args.epochs < 30:
        print(f"[!] Warning: Epochs ({args.epochs}) < 30. Enforcing minimum epochs = 30 per assignment requirement.")
        args.epochs = 30

    # Ensure output directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Device selection (CUDA if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Input Resolution: {args.img_width} x {args.img_height} (16:9 Aspect Ratio)")
    print(f"[*] Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print("=" * 65)

    # 1. Dataset & DataLoader (65% Train / 35% Test)
    train_ds, test_ds, _, _ = get_dataset_splits(
        dataset_dir=args.dataset_dir,
        train_ratio=0.65,
        seed=args.seed,
        img_size=(args.img_width, args.img_height),
        target_classes=[3],
        augment_train=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Model Initialization (From Scratch)
    model = CustomLaneSegNet(in_channels=3, num_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model successfully initialized from scratch. Total parameters: {total_params:,}")

    # 3. Loss & Optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. TensorBoard Integration
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"[*] TensorBoard initialized at: {args.log_dir}")

    # Log initial model graph to TensorBoard
    dummy_input = torch.randn(1, 3, args.img_height, args.img_width).to(device)
    try:
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"[*] Note: Model graph skipped: {e}")

    # 5. Training Loop
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, test_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        # TensorBoard Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Log sample prediction images every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(test_loader))
                sample_imgs = sample_batch['image'][:4].to(device)
                sample_masks = sample_batch['mask'][:4]
                sample_preds = torch.sigmoid(model(sample_imgs)).cpu() > 0.5
                
                # Stack image, ground truth, and prediction side by side
                for i in range(min(4, len(sample_imgs))):
                    writer.add_image(f'Sample_{i}/Input', sample_imgs[i].cpu(), epoch)
                    writer.add_image(f'Sample_{i}/GroundTruth', sample_masks[i], epoch)
                    writer.add_image(f'Sample_{i}/Prediction', sample_preds[i].float(), epoch)

        # Print Epoch Progress
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # Save Best Model Checkpoint based on Validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = os.path.join(args.checkpoint_dir, 'best_lane_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'img_size': (args.img_width, args.img_height)
            }, best_model_path)
            print(f"  --> Saved new best checkpoint (Val IoU: {best_val_iou:.4f}) to {best_model_path}")

    total_training_time = time.time() - start_time
    print("=" * 65)
    print(f"[*] Training Complete! Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} mins)")
    print(f"[*] Peak Validation IoU: {best_val_iou:.4f}")

    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_lane_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_iou': val_iou,
        'img_size': (args.img_width, args.img_height)
    }, final_model_path)
    print(f"[*] Final model checkpoint saved to: {final_model_path}")

    # Generate loss and IoU convergence graph
    plot_loss_curves(train_losses, val_losses, train_ious, val_ious, save_path='loss_convergence.png')
    writer.close()
    print("[*] TensorBoard logs finalized.")


if __name__ == '__main__':
    main()
