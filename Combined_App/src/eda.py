import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(data_dir="../data/raw", output_dir="../app/static"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Example structure parsing: assuming data_dir has subfolders for classes
    data = []
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # If no subdirectories, let's just parse all images
    if not classes:
        classes = ["All"]
        dirs_to_scan = [(data_dir, "All")]
    else:
        dirs_to_scan = [(os.path.join(data_dir, c), c) for c in classes]

    print("Extracting image metadata...")
    for folder, label in dirs_to_scan:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    if img is not None:
                        h, w, c = img.shape
                        data.append({
                            "filename": file,
                            "class": label,
                            "width": w,
                            "height": h,
                            "aspect_ratio": w / h,
                            "resolution": w * h,
                            "size_bytes": os.path.getsize(path),
                            "mean_r": img[:,:,2].mean(),
                            "mean_g": img[:,:,1].mean(),
                            "mean_b": img[:,:,0].mean()
                        })

    if not data:
        print("No images found for EDA.")
        return

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "dataset_stats.csv"), index=False)
    
    # 1. Class Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="class")
    plt.title("Class Distribution")
    plt.savefig(os.path.join(output_dir, "class_dist.png"))
    plt.close()
    
    # 2. Resolution Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="resolution", hue="class", kde=True)
    plt.title("Resolution Histogram")
    plt.savefig(os.path.join(output_dir, "resolution_hist.png"))
    plt.close()
    
    # 3. Aspect Ratio Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="aspect_ratio", hue="class", kde=True)
    plt.title("Aspect Ratio Histogram")
    plt.savefig(os.path.join(output_dir, "aspect_ratio_hist.png"))
    plt.close()
    
    print(f"EDA complete. Saved charts to {output_dir}")

if __name__ == "__main__":
    perform_eda()
