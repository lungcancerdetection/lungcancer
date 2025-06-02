import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import os
# --- 1. Define UNet++ Model ---
def conv_block(x, filters):
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    return x

def upsample_concat(x, skip):
    x = layers.UpSampling2D((2,2))(x)
    return layers.Concatenate()([x, skip])

def build_unetpp(input_shape):
    inputs = layers.Input(input_shape)
    filters = [32, 64, 128, 256, 512]

    ## Encoder
    x0_0 = conv_block(inputs, filters[0])
    p0 = layers.MaxPooling2D((2,2))(x0_0)

    x1_0 = conv_block(p0, filters[1])
    p1 = layers.MaxPooling2D((2,2))(x1_0)

    x2_0 = conv_block(p1, filters[2])
    p2 = layers.MaxPooling2D((2,2))(x2_0)

    x3_0 = conv_block(p2, filters[3])
    p3 = layers.MaxPooling2D((2,2))(x3_0)

    x4_0 = conv_block(p3, filters[4])

    ## Decoder with Nested Skip Connections
    x3_1 = conv_block(layers.Concatenate()([x3_0, upsample_concat(x4_0, x3_0)]), filters[3])
    x2_2 = conv_block(layers.Concatenate()([x2_0, upsample_concat(x3_1, x2_0)]), filters[2])
    x1_3 = conv_block(layers.Concatenate()([x1_0, upsample_concat(x2_2, x1_0)]), filters[1])
    x0_4 = conv_block(layers.Concatenate()([x0_0, upsample_concat(x1_3, x0_0)]), filters[0])

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(x0_4)

    model = Model(inputs, outputs)
    return model

# --- 2. Prepare model ---
input_shape = (256, 256, 1) # (change channels for your dataset)
model = build_unetpp(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Define paths
input_main_folder = ""
output_main_folder = ""

# Create output folder if it doesn't exist
os.makedirs(output_main_folder, exist_ok=True)

# Kernel for morphology
kernel = np.ones((5, 5), np.uint8)

# Traverse folders and images
for root, dirs, files in os.walk(input_main_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_image_path = os.path.join(root, file)

            # Read in grayscale
            image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue  # Skip corrupted or unreadable files

            # --- Step 1: Lung Segmentation ---
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
            morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            lung_mask = np.zeros_like(image)
            cv2.drawContours(lung_mask, contours, -1, 255, thickness=cv2.FILLED)
            segmented_lung = cv2.bitwise_and(image, image, mask=lung_mask)

            # --- Step 2: Disease Region Segmentation ---
            disease_mask = cv2.inRange(segmented_lung, 189, 245)
            disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            disease_region = cv2.bitwise_and(segmented_lung, segmented_lung, mask=disease_mask)

            # --- For visualization (disease overlay in red) ---
            segmented_lung_color = cv2.cvtColor(segmented_lung, cv2.COLOR_GRAY2BGR)
            segmented_lung_color[disease_mask > 0] = [255, 0, 0]  # Red overlay for disease

            # --- Step 3: Watershed Segmentation ---
            distance = cv2.distanceTransform(disease_mask, cv2.DIST_L2, 5)
            local_maxi = peak_local_max(distance, labels=disease_mask, footprint=np.ones((3, 3)), exclude_border=False)
            local_maxi_mask = np.zeros_like(distance, dtype=bool)
            if local_maxi.size > 0:
                local_maxi_mask[tuple(local_maxi.T)] = True
                markers = ndi.label(local_maxi_mask)[0]
                labels = watershed(-distance, markers, mask=disease_mask)
            else:
                labels = np.zeros_like(image)

            # --- Save Output Image ---
            relative_path = os.path.relpath(root, input_main_folder)
            output_folder = os.path.join(output_main_folder, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, f"watershed_{os.path.splitext(file)[0]}.png")

            plt.imsave(output_image_path, labels, cmap='nipy_spectral')

            # --- Final Visualization (optional, shown only) ---
            plt.figure(figsize=(20, 5))

            plt.subplot(1, 4, 1)
            plt.imshow(image, cmap='gray')
            plt.title("input image")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(disease_region, cmap='gray')
            plt.title("mask image")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(segmented_lung_color)
            plt.title("segmentation image")
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(labels, cmap='nipy_spectral')
            plt.title("Watershed Segmented Regions")
            plt.axis('off')

            plt.tight_layout()
            plt.show()