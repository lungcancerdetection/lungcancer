---------------Resized Image----------------
import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to your main input folder
input_folder = ''
output_folder = ''

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop over each subfolder
for root, dirs, files in os.walk(input_folder):
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if image_files:
        # Print folder name
        folder_name = os.path.basename(root)
        print(f"Showing and saving images from folder: {folder_name}")

        # Create a subfolder in the output directory for each input subfolder
        folder_output_path = os.path.join(output_folder, folder_name)
        os.makedirs(folder_output_path, exist_ok=True)

        for file in image_files:
            img_path = os.path.join(root, file)

            # Open the image
            img = Image.open(img_path).convert('RGB')

            # Resize to 256x256
            img_resized = img.resize((256, 256))

            # Display Original and Resized Image side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Original Image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Resized Image
            axes[1].imshow(img_resized)
            axes[1].set_title('Resized Image (256x256)')
            axes[1].axis('off')

            plt.suptitle(f"Folder: {folder_name} | Image: {file}")
            plt.show()

            # Save resized image
            resized_img_path = os.path.join(folder_output_path, f"resized_{file}")
            img_resized.save(resized_img_path)
            print(f"Saved resized image: {resized_img_path}")
----------clahe_equalize_image---------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def(image):
    """
    Apply CLAHE to a color image using LAB color space.
    """
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    equalized_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return equalized_image

def display_images(original_image, equalized_image):
    """
    Display original and CLAHE-enhanced images side by side.
    """
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    equalized_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("CLAHE Enhanced Image")
    plt.imshow(equalized_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_and_equalize_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image types
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is not None:
                    # Apply CLAHE enhancement
                    equalized_image = clahe_equalize_image(image)

                    # Display original and enhanced image
                    display_images(image, equalized_image)

                    # Generate output subfolder path based on input folder structure
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Save the CLAHE-enhanced image in the output folder
                    output_image_path = os.path.join(output_subfolder, filename)
                    cv2.imwrite(output_image_path, equalized_image)
                    print(f"CLAHE enhanced and saved: {output_image_path}")
                else:
                    print(f"Failed to load: {filename}")

# === USAGE ===
input_folder = ''  # Input folder with multiple subfolders
output_folder = ''  # Output folder

process_and_equalize_images(input_folder, output_folder)
print("CLAHE enhancement and display completed.")
--------non_local_means_denoising-------------------------------
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def non_local_means_denoising(image, h=10, templateWindowSize=7, searchWindowSize=21):
    """
    Apply Non-Local Means denoising to the image using OpenCV.
    """
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, templateWindowSize, searchWindowSize)
    return denoised_image

def display_images(original_image, denoised_image):
    """
    Display original and denoised images side by side for comparison.
    """
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image (Non-Local Means)")
    plt.imshow(denoised_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_and_denoise_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image types
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is not None:
                    # Apply Non-Local Means denoising
                    denoised_image = non_local_means_denoising(image)

                    # Display original and denoised image
                    display_images(image, denoised_image)

                    # Generate output subfolder path based on input folder structure
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Save the denoised image in the output folder
                    output_image_path = os.path.join(output_subfolder, filename)
                    cv2.imwrite(output_image_path, denoised_image)
                    print(f"Denoised image saved: {output_image_path}")
                else:
                    print(f"Failed to load: {filename}")

# === USAGE ===
input_folder = ''  # Input folder with multiple subfolders
output_folder = ''  # Output folder

process_and_denoise_images(input_folder, output_folder)
print("Non-Local Means denoising and display completed.")
----------min-max normalization-----------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def minmax_normalize_image(image):
    """
    Apply Min-Max normalization to an image (per-image basis).
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def display_images(original_image, normalized_image):
    """
    Display original and normalized images side by side.
    """
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    normalized_rgb = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Min-Max Normalized Image")
    plt.imshow(normalized_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_and_normalize_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is not None:
                    normalized_image = minmax_normalize_image(image)
                    display_images(image, normalized_image)

                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder, filename)
                    cv2.imwrite(output_image_path, normalized_image)
                    print(f"Normalized and saved: {output_image_path}")
                else:
                    print(f"Failed to load: {filename}")

# === USAGE ===
input_folder = ''
output_folder = ''

process_and_normalize_images(input_folder, output_folder)
print("Normalization and display completed.")