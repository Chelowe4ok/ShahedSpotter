import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

# --- PATH CONFIGURATION ---
# Resolve the absolute path to the data/ folder (where this script lives)
CURRENT_DIR = Path(__file__).parent

# Build paths relative to the data/ folder
IMAGES_DIR = CURRENT_DIR / "train" / "images"
LABELS_DIR = CURRENT_DIR / "train" / "labels"

def analyze_dataset(images_dir, labels_dir):
    # Lists for collecting per-image and per-object metrics
    object_areas = []          # Absolute object area in pixels
    image_brightness = []      # Mean frame brightness
    image_contrast = []        # Frame contrast (std dev of grayscale)
    background_brightness = [] # Background brightness (excluding object regions)
    image_blur_scores = []     # Sharpness score — higher means sharper

    image_paths = list(images_dir.glob("*.jpg")) # Or *.png
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Analysing {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        # 1. Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Domain diversity — frame brightness and contrast
        img_mean = np.mean(gray)
        img_std = np.std(gray)
        image_brightness.append(img_mean)
        image_contrast.append(img_std)

        # 3. Sharpness estimate via Laplacian variance
        # Values below 100 are generally considered blurry
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        image_blur_scores.append(laplacian_var)

        # 4. Object size and background brightness analysis
        label_path = labels_dir / (img_path.stem + ".txt")

        # Background mask — starts as all-True (full frame is background)
        bg_mask = np.ones((h, w), dtype=bool)

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])

                        # Convert normalised YOLO coordinates to absolute pixels
                        abs_w = int(bbox_w * w)
                        abs_h = int(bbox_h * h)
                        area = abs_w * abs_h
                        object_areas.append(area)

                        # Mask out the object region from the background
                        x1 = max(0, int((x_center - bbox_w/2) * w))
                        y1 = max(0, int((y_center - bbox_h/2) * h))
                        x2 = min(w, x1 + abs_w)
                        y2 = min(h, y1 + abs_h)
                        bg_mask[y1:y2, x1:x2] = False

        # Compute sky/background brightness excluding drone and bird regions
        if np.any(bg_mask):
            bg_mean = np.mean(gray[bg_mask])
            background_brightness.append(bg_mean)

    # --- PLOTS ---
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO Dataset Analysis: ShahedSpotter', fontsize=16)

    # Plot 1: Absolute object area distribution
    axs[0, 0].hist(object_areas, bins=50, color='royalblue', edgecolor='black')
    axs[0, 0].set_title('Absolute Object Area (sq. pixels)')
    axs[0, 0].set_xlabel('Area (Width x Height)')
    axs[0, 0].set_ylabel('Object count')
    # Clip X axis at the 95th percentile to suppress outlier influence
    if object_areas:
        axs[0, 0].set_xlim(0, np.percentile(object_areas, 95))

    # Plot 2: Background brightness distribution
    axs[0, 1].hist(background_brightness, bins=30, color='orange', edgecolor='black')
    axs[0, 1].set_title('Domain: Background Brightness (0=Night, 255=Bright sky)')
    axs[0, 1].set_xlabel('Mean background pixel value')
    axs[0, 1].set_ylabel('Image count')

    # Plot 3: Frame contrast distribution
    axs[1, 0].hist(image_contrast, bins=30, color='mediumseagreen', edgecolor='black')
    axs[1, 0].set_title('Domain: Contrast (Standard Deviation)')
    axs[1, 0].set_xlabel('Contrast value (higher = more contrast)')
    axs[1, 0].set_ylabel('Image count')

    # Plot 4: Sharpness (blur) distribution
    axs[1, 1].hist(image_blur_scores, bins=50, color='tomato', edgecolor='black')
    axs[1, 1].set_title('Sharpness: Laplacian Variance (< 100 = blurry)')
    axs[1, 1].set_xlabel('Sharpness score')
    axs[1, 1].set_ylabel('Image count')
    if image_blur_scores:
        axs[1, 1].set_xlim(0, np.percentile(image_blur_scores, 90))  # Drop extreme sharpness outliers

    plt.tight_layout()
    plt.savefig('dataset_analysis_report.png', dpi=300)
    plt.show()

    # Text summary
    print("\n=== SUMMARY REPORT ===")
    if object_areas:
        print(f"Mean object area  : {np.mean(object_areas):.0f} px")
        print(f"Median object area: {np.median(object_areas):.0f} px  (if < 1000 px, objects are VERY small)")

    blur_threshold = 100
    blurry_images = sum(1 for score in image_blur_scores if score < blur_threshold)
    print(f"Blurry images (<100 Laplacian): {blurry_images} of {len(image_paths)} ({(blurry_images/len(image_paths))*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset(IMAGES_DIR, LABELS_DIR)
