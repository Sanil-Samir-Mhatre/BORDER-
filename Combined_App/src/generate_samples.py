import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_document(filename, is_forged=False):
    # Create a blank white image (aspect ratio similar to passport)
    img = Image.new('RGB', (600, 400), color='white')
    d = ImageDraw.Draw(img)
    
    # Draw some "document" features
    d.rectangle([10, 10, 590, 390], outline="black", width=2)
    d.rectangle([20, 50, 150, 200], outline="gray", width=2) # Photo box
    d.text((30, 20), "PASSPORT", fill="black")
    
    # Add text
    d.text((170, 60), "Name: JOHN DOE", fill="black")
    d.text((170, 90), "Nationality: IND", fill="black")
    
    dob = "01/01/1980"
    mrz_dob = "800101"
    
    if is_forged:
        dob = "05/05/1990" # Mismatch
        # simulate some copy-move noise
        img.paste(img.crop((170, 60, 300, 80)), (170, 120))
        
    d.text((170, 120), f"DOB: {dob}", fill="black")
    d.text((170, 150), "Sex: M", fill="black")
    
    # Draw mock MRZ
    d.text((20, 320), f"P<INDDOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<", fill="black")
    d.text((20, 350), f"A1234567<8IND{mrz_dob}3M3001018<<<<<<<<<<<<<<0", fill="black")
    
    img.save(filename)

if __name__ == "__main__":
    os.makedirs("../sample_inputs", exist_ok=True)
    create_synthetic_document("../sample_inputs/genuine_sample.jpg", is_forged=False)
    create_synthetic_document("../sample_inputs/forged_sample.jpg", is_forged=True)
    print("Created sample inputs.")
