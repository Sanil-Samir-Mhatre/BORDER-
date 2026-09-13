import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Import local modules
from src.models import build_efficientnet, get_random_forest, EvidenceFusion
from src.forensics import extract_forensic_features

# Mock Dataset class
class DocumentDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def evaluate_model(y_true, y_pred, y_prob, model_name, output_dir):
    """
    Generate metrics and plots for a model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
        
    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Forged'], yticklabels=['Genuine', 'Forged'])
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_cm.png"))
    plt.close()
    
    return metrics

def run_training_pipeline(data_dir="../data/raw", output_dir="../app/static"):
    print("Starting training pipeline...")
    
    # For a real run, we'd load paths from data_dir.
    # We'll mock a small subset of results for demonstration if no images are found yet.
    # In a full run, we would loop over all images and train.
    
    # Mocking results for the prototype dashboard
    print("Generating evaluation metrics and charts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate fake true labels and predictions to demonstrate charts
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    
    # Random Forest (decent but struggles with complex patterns)
    rf_prob = np.clip(y_true * 0.7 + np.random.normal(0.2, 0.2, n_samples), 0, 1)
    rf_pred = (rf_prob > 0.5).astype(int)
    rf_metrics = evaluate_model(y_true, rf_pred, rf_prob, "Random Forest", output_dir)
    
    # EfficientNet (better on image anomalies)
    cnn_prob = np.clip(y_true * 0.85 + np.random.normal(0.1, 0.1, n_samples), 0, 1)
    cnn_pred = (cnn_prob > 0.5).astype(int)
    cnn_metrics = evaluate_model(y_true, cnn_pred, cnn_prob, "EfficientNet", output_dir)
    
    # Fusion (best overall)
    fusion_prob = 0.6 * cnn_prob + 0.4 * rf_prob
    fusion_pred = (fusion_prob > 0.5).astype(int)
    fusion_metrics = evaluate_model(y_true, fusion_pred, fusion_prob, "Evidence Fusion", output_dir)
    
    # Save comparison table
    df_metrics = pd.DataFrame([rf_metrics, cnn_metrics, fusion_metrics])
    df_metrics.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # --- SAVE MODEL FILES ---
    # Instantiate the models from src.models
    print("Saving model weights and artifacts...")
    os.makedirs("../models", exist_ok=True)
    
    # Save Random Forest
    import joblib
    rf_model = get_random_forest()
    # Mock training on some random features for saving purposes
    X_mock = np.random.rand(10, 5)
    y_mock = np.random.randint(0, 2, 10)
    rf_model.fit(X_mock, y_mock)
    joblib.dump(rf_model, "../models/random_forest.pkl")
    
    # Save EfficientNet architecture
    cnn_model = build_efficientnet(num_classes=2)
    # Save the untrained state_dict just to provide the model artifact for the demo
    torch.save(cnn_model.state_dict(), "../models/efficientnet_b0.pth")
    
    print("Training pipeline simulation complete. Models saved to 'models/' directory.")
    
if __name__ == "__main__":
    run_training_pipeline()
