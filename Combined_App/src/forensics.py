import cv2
import numpy as np
import os

def calculate_ela(image, quality=90):
    """
    Generate Error Level Analysis (ELA) map.
    Saves a temporary JPEG to calculate differences.
    """
    tmp_filename = "temp_ela.jpg"
    cv2.imwrite(tmp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_image = cv2.imread(tmp_filename)
    os.remove(tmp_filename)
    
    # Calculate absolute difference
    diff = cv2.absdiff(image, compressed_image)
    
    # Scale differences to be visible
    max_diff = np.max(diff)
    if max_diff == 0:
        max_diff = 1
    
    ela_map = (diff * (255.0 / max_diff)).astype(np.uint8)
    return ela_map

def extract_ela_stats(ela_map):
    """
    Extract statistical features from ELA map.
    """
    gray = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)
    return {
        "ela_mean": np.mean(gray),
        "ela_std": np.std(gray),
        "ela_max": np.max(gray)
    }

def detect_copy_move(image):
    """
    Detect copy-move forgery using ORB.
    Returns a score indicating how many dense matches exist.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None or len(descriptors) < 2:
        return 0
        
    # Match descriptors to themselves
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=2)
    
    # Filter matches using Lowe's ratio test and removing self-matches
    good_matches = []
    for m in matches:
        if len(m) == 2:
            # m[0] is self, m[1] is the next best match
            if m[1].distance < 0.75 * m[0].distance: # Ratio test
                pt1 = np.array(keypoints[m[0].queryIdx].pt)
                pt2 = np.array(keypoints[m[1].trainIdx].pt)
                # Check spatial distance (should be far apart to be copy-move)
                if np.linalg.norm(pt1 - pt2) > 50:
                    good_matches.append(m[1])
                    
    return len(good_matches)

def extract_forensic_features(image):
    """
    Combine all forensic features.
    """
    ela_map = calculate_ela(image)
    ela_stats = extract_ela_stats(ela_map)
    copy_move_score = detect_copy_move(image)
    
    return {
        **ela_stats,
        "copy_move_score": copy_move_score,
        "blur_score": cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    }
