import cv2
import numpy as np

def detect_boundary_and_crop(image):
    """
    Detect document boundary and crop it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image # return original if no boundary found
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Check if the cropped region is reasonably large, otherwise it's just noise
    if w * h > 0.3 * image.shape[0] * image.shape[1]:
        return image[y:y+h, x:x+w]
    return image

def deskew(image):
    """
    Deskew using Hough line transform.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # we only care about small skews for documents
            if -45 < angle < 45:
                angles.append(angle)
                
        if angles:
            median_angle = np.median(angles)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    return image

def denoise(image):
    """
    Denoise using Bilateral filter.
    """
    return cv2.bilateralFilter(image, 9, 75, 75)

def enhance_contrast(image):
    """
    Contrast enhancement using CLAHE.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def resize_image(image, size=(224, 224)):
    """
    Resize image to input size for CNN.
    """
    return cv2.resize(image, size)

def preprocess_pipeline(image):
    """
    Run the full preprocessing pipeline.
    """
    cropped = detect_boundary_and_crop(image)
    deskewed = deskew(cropped)
    denoised = denoise(deskewed)
    enhanced = enhance_contrast(denoised)
    final = resize_image(enhanced)
    
    # Return intermediate steps for visualization as requested in prompt
    return {
        "original": image,
        "cropped": cropped,
        "deskewed": deskewed,
        "denoised": denoised,
        "enhanced": enhanced,
        "final": final
    }
