import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import preprocess_pipeline
from eda import perform_eda
from forensics import calculate_ela

def save_image(img, path):
    if len(img.shape) == 3: # BGR
        cv2.imwrite(path, img)
    else: # Grayscale
        cv2.imwrite(path, img)

def generate_preprocessing_images(input_path, output_prefix):
    os.makedirs("../app/static", exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read {input_path}")
        return
        
    save_image(img, f"../app/static/{output_prefix}_before_preprocessing.jpg")
    
    stages = preprocess_pipeline(img)
    save_image(stages["final"], f"../app/static/{output_prefix}_after_preprocessing.jpg")
    print(f"Saved preprocessing images for {output_prefix}")

def generate_forensic_images(input_path, output_prefix):
    os.makedirs("../app/static", exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        return
        
    ela_map = calculate_ela(img)
    save_image(ela_map, f"../app/static/{output_prefix}_ela_map.jpg")
    print(f"Saved forensic features image for {output_prefix}")

if __name__ == "__main__":
    genuine_path = "../sample_inputs/genuine_sample.jpg"
    forged_path = "../sample_inputs/forged_sample.jpg"
    
    print("Generating preprocessing images...")
    generate_preprocessing_images(genuine_path, "genuine")
    generate_preprocessing_images(forged_path, "forged")
    
    print("Generating feature extraction images (ELA)...")
    generate_forensic_images(genuine_path, "genuine")
    generate_forensic_images(forged_path, "forged")
    
    print("Generating EDA images...")
    # This might fail if the folder structure is not correct, but since eda handles flat dirs, we'll see
    perform_eda(data_dir="../sample_inputs", output_dir="../app/static")
    print("All image generation complete.")
