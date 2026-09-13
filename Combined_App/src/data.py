import kagglehub
import os
import shutil

def download_dataset(download_dir="../data/raw"):
    """
    Downloads the passport dataset using kagglehub.
    Returns the path to the downloaded files.
    """
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("simongraves/passport-dataset")
    print(f"Dataset downloaded to original path: {path}")
    
    # We will copy it to our local data/raw folder
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Copying files to local directory: {download_dir}")
    # Use shutil to copy tree if it's a directory
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(download_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.copytree(s, d)
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)
                
    print("Dataset ready in local folder.")
    return download_dir

if __name__ == "__main__":
    download_dataset()
