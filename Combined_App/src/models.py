import torch
import torch.nn as nn
from torchvision import models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

def build_efficientnet(num_classes=2):
    """
    Build EfficientNetB0 pretrained on ImageNet.
    Modifies the final layer for binary classification (Genuine vs Forged).
    """
    # Using torchvision models
    model = models.efficientnet_b0(pretrained=True)
    
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_random_forest():
    """
    Returns a configured Random Forest for classical ML.
    """
    return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

def get_xgboost():
    """
    Returns configured XGBoost model if available.
    """
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

class EvidenceFusion(nn.Module):
    """
    Fuses CNN output (Model B) with Classical output (Model A) or features.
    In this prototype, we'll implement a simple soft-voting or rule-based fusion at inference.
    """
    def __init__(self, cnn_weight=0.6, rf_weight=0.4):
        super().__init__()
        self.cnn_weight = cnn_weight
        self.rf_weight = rf_weight

    def predict_proba(self, cnn_probs, rf_probs):
        """
        Takes probability arrays from both models and fuses them.
        """
        # Assuming index 1 is 'forged' probability
        return self.cnn_weight * cnn_probs + self.rf_weight * rf_probs

