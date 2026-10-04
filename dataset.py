"""
dataset.py - Dataset Loading, Preprocessing, and Augmentation for Lane Segmentation
PSU-reservoir Lane Dataset (1,000 frames, original 1280x720 RGB).

Features:
    - Extracts polygon coordinates from YOLO segmentation annotations
    - Rasterizes polygons into Binary Masks using cv2.fillPoly
    - Configurable target classes (default class 3: 'lane')
    - Resizes to low-memory footprint dimensions (e.g., 128x72, 64x36)
    - Reproducible 65% Train / 35% Test split (650 train / 350 test)
    - Synchronized Data Augmentation (Flip, Rotation, Brightness/Contrast)
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


def load_polygon_mask(label_path: str, orig_w: int, orig_h: int, target_classes: List[int] = [3]) -> np.ndarray:
    """
    Reads YOLO segmentation polygon file and renders binary mask of shape (orig_h, orig_w).
    Polygon coordinates in label are normalized (0.0 - 1.0).
    """
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in target_classes:
                coords = [float(x) for x in parts[1:]]
                if len(coords) >= 6:  # At least 3 points
                    pts = np.array(
                        [(int(coords[i] * orig_w), int(coords[i + 1] * orig_h)) for i in range(0, len(coords), 2)],
                        dtype=np.int32
                    )
                    cv2.fillPoly(mask, [pts], 1)
    return mask


class LaneDataset(Dataset):
    """
    PyTorch Dataset for PSU-reservoir Lane Segmentation.
    """
    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        img_size: Tuple[int, int] = (64, 36),  # (width, height) - default 16:9 scale per lecture notes
        target_classes: List[int] = [3],
        is_train: bool = False,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_w, self.img_h = img_size
        self.target_classes = target_classes
        self.is_train = is_train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        # 1. Read Image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Render Binary Mask at original resolution (1280x720)
        mask_orig = load_polygon_mask(lbl_path, orig_w, orig_h, self.target_classes)

        # 3. Resize to model input resolution
        img_resized = cv2.resize(img_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_orig, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # 4. Data Augmentation (Train set only)
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_resized = np.fliplr(img_resized).copy()
                mask_resized = np.fliplr(mask_resized).copy()

            # Random Brightness & Contrast adjustment
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-20, 20)    # Brightness
                img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

            # Random small rotation (+/- 5 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-5.0, 5.0)
                center = (self.img_w / 2.0, self.img_h / 2.0)
                rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
                img_resized = cv2.warpAffine(img_resized, rot_mat, (self.img_w, self.img_h), flags=cv2.INTER_LINEAR)
                mask_resized = cv2.warpAffine(mask_resized, rot_mat, (self.img_w, self.img_h), flags=cv2.INTER_NEAREST)

        # 5. Convert to PyTorch Tensors
        # Image: [C, H, W] normalized to [0.0, 1.0]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # Mask: [1, H, W] float32 {0.0, 1.0}
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'image_path': img_path,
            'label_path': lbl_path,
            'orig_size': (orig_w, orig_h),
            'filename': os.path.basename(img_path)
        }


def get_dataset_splits(
    dataset_dir: str = 'lane_dataset/dataset_seg',
    train_ratio: float = 0.65,
    seed: int = 42,
    img_size: Tuple[int, int] = (64, 36),
    target_classes: List[int] = [3],
    augment_train: bool = True,
) -> Tuple[LaneDataset, LaneDataset, List[str], List[str]]:
    """
    Creates reproducible 65% Train / 35% Test datasets according to specification.
    Returns (train_dataset, test_dataset, train_files, test_files).
    """
    img_dir = os.path.join(dataset_dir, 'images', 'all_images')
    lbl_dir = os.path.join(dataset_dir, 'labels', 'all_images')

    all_lbls = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    
    paired_images = []
    paired_labels = []

    for lbl_name in all_lbls:
        img_name = os.path.splitext(lbl_name)[0] + '.jpg'
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)
        if os.path.exists(img_path):
            paired_images.append(img_path)
            paired_labels.append(lbl_path)

    total_samples = len(paired_images)
    print(f"[*] Found {total_samples} valid image-label pairs.")

    # Reproducible shuffle
    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(total_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_imgs = [paired_images[i] for i in train_indices]
    train_lbls = [paired_labels[i] for i in train_indices]
    test_imgs = [paired_images[i] for i in test_indices]
    test_lbls = [paired_labels[i] for i in test_indices]

    print(f"[*] Split summary: Train = {len(train_imgs)} ({train_ratio*100:.1f}%), Test = {len(test_imgs)} ({(1-train_ratio)*100:.1f}%)")

    # Save split file lists for predict.py and evaluation.py
    os.makedirs('splits', exist_ok=True)
    with open('splits/train_files.txt', 'w', encoding='utf-8') as f:
        for p in train_imgs:
            f.write(f"{os.path.basename(p)}\n")
    with open('splits/test_files.txt', 'w', encoding='utf-8') as f:
        for p in test_imgs:
            f.write(f"{os.path.basename(p)}\n")

    train_dataset = LaneDataset(
        train_imgs,
        train_lbls,
        img_size=img_size,
        target_classes=target_classes,
        is_train=True,
        augment=augment_train,
    )

    test_dataset = LaneDataset(
        test_imgs,
        test_lbls,
        img_size=img_size,
        target_classes=target_classes,
        is_train=False,
        augment=False,
    )

    return train_dataset, test_dataset, train_imgs, test_imgs


if __name__ == '__main__':
    train_ds, test_ds, _, _ = get_dataset_splits(img_size=(128, 72))
    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Test Dataset Size:  {len(test_ds)}")

    sample = train_ds[0]
    print(f"Sample Image Tensor Shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
    print(f"Sample Mask Tensor Shape:  {sample['mask'].shape}, min: {sample['mask'].min()}, max: {sample['mask'].max()}")
    print("Dataset verification successful!")
