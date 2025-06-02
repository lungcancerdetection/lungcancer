import cv2
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis

# Main folder containing subfolders of images
input_main_folder = ''

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Shape Features
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_area = 0
    shape_eccentricity = 0
    shape_roundness = 0

    if contours:
        contour = max(contours, key=cv2.contourArea)  # Use largest contour
        area = cv2.contourArea(contour)
        if len(contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0
        else:
            eccentricity = 0
        perimeter = cv2.arcLength(contour, True)
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        shape_area = area
        shape_eccentricity = eccentricity
        shape_roundness = roundness

    # Texture Features
    glcm_contrast = np.var(image)
    glcm_dissimilarity = np.mean(image)
    glcm_homogeneity = np.std(image)

    # Local Binary Pattern (LBP) Features
    radius = 1
    n_points = 8 * radius
    lbp = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            binary_string = ''
            for k in range(8):
                dx = int(np.cos(np.pi * k / 4) * radius)
                dy = int(np.sin(np.pi * k / 4) * radius)
                binary_string += '1' if image[i + dx, j + dy] > center else '0'
            lbp[i, j] = int(binary_string, 2)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Statistical Texture Features
    mean_intensity = np.mean(image)
    variance_intensity = np.var(image)
    skewness_intensity = skew(image.ravel(), nan_policy='omit')
    kurtosis_intensity = kurtosis(image.ravel(), nan_policy='omit')
    entropy = -np.sum((image / 255.0) * np.log2((image / 255.0) + 1e-9))

    return {
        'mean_intensity': mean_intensity,
        'variance_intensity': variance_intensity,
        'skewness_intensity': skewness_intensity,
        'kurtosis_intensity': kurtosis_intensity,
        'entropy': entropy,
        'glcm_contrast': glcm_contrast,
        'glcm_dissimilarity': glcm_dissimilarity,
        'glcm_homogeneity': glcm_homogeneity,
        'lbp_hist': lbp_hist.tolist(),
        'shape_area': shape_area,
        'shape_eccentricity': shape_eccentricity,
        'shape_roundness': shape_roundness,
    }

# Traverse all images and extract features with labels
features_list = []

for root, dirs, files in os.walk(input_main_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_main_folder)
            label = os.path.basename(relative_path)

            features = extract_features(image_path)
            features['image'] = file
            features['label'] = label
            features_list.append(features)

# Create DataFrame
df = pd.DataFrame(features_list)

# Fill NaNs (if any) with zeros
df = df.fillna(0)

# Output CSV path
output_csv_path = ''
df.to_csv(output_csv_path, index=False)

print(f"Features with labels saved to {output_csv_path}")
df