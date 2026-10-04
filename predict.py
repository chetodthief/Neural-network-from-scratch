"""
predict.py - Inference Pipeline for Custom Lane Segmentation Neural Network
Features:
    - Loads trained model checkpoint
    - Evaluates on 35% Test Set (350 frames) or single image
    - Resizes predicted mask back to original resolution (1280x720 px)
    - Saves binary masks (0/255 PNG) and visual overlay snapshots
    - Accurately measures Memory Footprint (RAM & GPU VRAM) and Latency (FPS)
"""

import os
import time
import argparse
import psutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import CustomLaneSegNet


def get_memory_footprint(device: torch.device) -> dict:
    """Measures current system RAM and GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    vram_alloc_mb = 0.0
    vram_peak_mb = 0.0
    if device.type == 'cuda':
        vram_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        'ram_mb': ram_mb,
        'vram_alloc_mb': vram_alloc_mb,
        'vram_peak_mb': vram_peak_mb,
    }


def predict_single_image(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    img_size: tuple,
    device: torch.device,
    threshold: float = 0.5
) -> tuple:
    """
    Runs model inference on a single image and resizes binary mask back to original resolution.
    Returns:
        mask_original_res: np.ndarray uint8 of shape (orig_h, orig_w) with values {0, 255}
        prob_map_original: np.ndarray float32 of shape (orig_h, orig_w) with values [0.0, 1.0]
        inference_time_ms: float
    """
    orig_h, orig_w = img_bgr.shape[:2]
    img_w, img_h = img_size

    # Preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device)

    # Inference with timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    inf_time_ms = (t_end - t_start) * 1000.0

    # Resize probability map and mask back to original resolution (1280x720)
    prob_map_original = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map_original >= threshold).astype(np.uint8) * 255

    return binary_mask, prob_map_original, inf_time_ms


def create_overlay(img_bgr: np.ndarray, binary_mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    """Blends a colored mask onto the original RGB image."""
    overlay = img_bgr.copy()
    mask_indices = binary_mask > 0
    overlay[mask_indices] = color
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)
    return blended


def main():
    parser = argparse.ArgumentParser(description="Lane Segmentation Inference and Mask Generation")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_lane_model.pth', help="Path to checkpoint")
    parser.add_argument('--dataset-dir', type=str, default='lane_dataset/dataset_seg', help="Path to dataset_seg")
    parser.add_argument('--split-file', type=str, default='splits/test_files.txt', help="List of test images (35% split)")
    parser.add_argument('--output-dir', type=str, default='output', help="Output root directory")
    parser.add_argument('--threshold', type=float, default=0.5, help="Binary classification threshold")
    parser.add_argument('--device', type=str, default='', help="Force 'cuda' or 'cpu'")
    parser.add_argument('--single-image', type=str, default='', help="Optional single image path to test")
    parser.add_argument('--save-overlays', action='store_true', default=True, help="Save visual blend overlay images")
    parser.add_argument('--save-max-samples', type=int, default=350, help="Max test samples to process")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print(f"[*] Lane Segmentation Inference Pipeline")
    print(f"[*] Inference Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"[*] Checkpoint:       {args.checkpoint}")
    print(f"[*] Output Directory: {args.output_dir}")
    print("=" * 65)

    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Please run train.py first.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    img_size = checkpoint.get('img_size', (128, 72))

    # 2. Instantiate Model and Load Weights
    model = CustomLaneSegNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[*] Model loaded successfully (Input size: {img_size[0]}x{img_size[1]}).")

    # Output subdirectories
    masks_dir = os.path.join(args.output_dir, 'predictions_mask')
    overlays_dir = os.path.join(args.output_dir, 'predictions_overlay')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    # 3. Determine images to process
    if args.single_image:
        test_images = [args.single_image]
    else:
        # Load from test split file (35% test set)
        if not os.path.exists(args.split_file):
            raise FileNotFoundError(f"Test split file not found: {args.split_file}. Run train.py or dataset.py first.")
        with open(args.split_file, 'r', encoding='utf-8') as f:
            filenames = [line.strip() for line in f if line.strip()]
        img_dir = os.path.join(args.dataset_dir, 'images', 'all_images')
        test_images = [os.path.join(img_dir, fname) for fname in filenames[:args.save_max_samples]]

    print(f"[*] Total test frames to process: {len(test_images):,}")

    # Warmup GPU
    dummy_warmup = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        _ = predict_single_image(model, dummy_warmup, img_size, device, args.threshold)

    # 4. Run Inference on Test Set
    inference_times = []
    initial_mem = get_memory_footprint(device)

    for idx, img_path in enumerate(test_images):
        fname = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[!] Warning: Unable to read {img_path}")
            continue

        binary_mask, prob_map, inf_time_ms = predict_single_image(
            model, img_bgr, img_size, device, args.threshold
        )
        inference_times.append(inf_time_ms)

        # Save Binary Mask (1280x720)
        base_name = os.path.splitext(fname)[0]
        mask_out_path = os.path.join(masks_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_out_path, binary_mask)

        # Save Visualization Overlay
        if args.save_overlays:
            overlay_img = create_overlay(img_bgr, binary_mask, color=(0, 255, 0), alpha=0.40)
            overlay_out_path = os.path.join(overlays_dir, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_out_path, overlay_img)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_images):
            print(f"  --> Processed {idx + 1}/{len(test_images)} frames | Latency: {np.mean(inference_times[-50:]):.2f} ms")

    # 5. Measure Final Memory Footprint & Statistics
    final_mem = get_memory_footprint(device)
    avg_latency_ms = np.mean(inference_times)
    fps = 1000.0 / avg_latency_ms

    print("\n" + "=" * 65)
    print("INFERENCE BENCHMARK & MEMORY FOOTPRINT REPORT")
    print("=" * 65)
    print(f"Total Test Images Processed: {len(test_images)}")
    print(f"Average Inference Latency:   {avg_latency_ms:.2f} ms / frame")
    print(f"Inference Throughput (FPS):  {fps:.1f} frames / second")
    print(f"Process RAM Footprint (RSS): {final_mem['ram_mb']:.2f} MB")
    if device.type == 'cuda':
        print(f"GPU VRAM Allocated:          {final_mem['vram_alloc_mb']:.2f} MB")
        print(f"GPU VRAM Peak (Max):         {final_mem['vram_peak_mb']:.2f} MB")
    print(f"Output Binary Masks:         {masks_dir}")
    print(f"Output Overlay Images:       {overlays_dir}")
    print("=" * 65)

    # Save benchmark stats to text file
    with open('output/inference_benchmark.txt', 'w', encoding='utf-8') as f:
        f.write("Inference Benchmark & Memory Footprint Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Device:                      {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})\n")
        f.write(f"Input Resolution:            {img_size[0]} x {img_size[1]}\n")
        f.write(f"Original Resolution:         1280 x 720\n")
        f.write(f"Total Frames Tested:         {len(test_images)}\n")
        f.write(f"Average Latency:             {avg_latency_ms:.2f} ms\n")
        f.write(f"Throughput:                  {fps:.1f} FPS\n")
        f.write(f"Host RAM Footprint:          {final_mem['ram_mb']:.2f} MB\n")
        if device.type == 'cuda':
            f.write(f"GPU VRAM Allocated:          {final_mem['vram_alloc_mb']:.2f} MB\n")
            f.write(f"GPU VRAM Peak:               {final_mem['vram_peak_mb']:.2f} MB\n")


if __name__ == '__main__':
    main()
