import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0
from tensorflow.keras.utils import to_categorical
import cv2
import os

# --------------------- Step 1: Load and preprocess tabular data ---------------------
csv_file = ""
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

if 'label' not in df.columns:
    raise ValueError("Column 'label' not found in the dataset.")

df = df.drop(columns=[col for col in df.columns if df[col].dtype == 'object' and col != 'label'], errors="ignore")
X = df.drop(columns=["label"], errors="ignore").values
y = df["label"].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle inf and nan
X = np.array(X, dtype=np.float32)
X = np.where(np.isinf(X), np.nan, X)
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Determine number of classes
num_classes = len(np.unique(y))

# Reshape for CNN: simulate 2D image from feature vector
X_reshaped = np.array([cv2.resize(np.tile(x.reshape(1, -1), (224, 1)), (224, 224)) for x in X])
X_reshaped = np.stack([np.stack([img] * 3, axis=-1) for img in X_reshaped])  # Make 3-channel

# One-hot encode labels
y_categorical = to_categorical(y, num_classes=num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# --------------------- Step 2: Squeeze-and-Excitation Block ---------------------
def squeeze_excite_block(input, ratio=8):
    filters = input.shape[-1]
    se = layers.GlobalAveragePooling2D()(input)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.multiply([input, se])

# --------------------- Step 3: Custom Lightweight CNN ---------------------
def custom_lightweight_cnn(input_shape, num_classes):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same')(input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    for _ in range(2):
        x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
        x = squeeze_excite_block(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=input, outputs=output)

# --------------------- Step 4: Pretrained Models ---------------------
def create_mobilenetv3_model(input_shape, num_classes):
    base = MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base.output)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=output)

def create_efficientnetb0_model(input_shape, num_classes):
    base = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base.output)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=output)

def create_shufflenetv2_substitute(input_shape, num_classes):
    base = MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
    x = layers.GlobalAveragePooling2D()(base.output)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=output)

# --------------------- Step 5: Train Model ---------------------
def compile_and_train_model(model, X_train, y_train, epochs=10):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

input_shape = (224, 224, 3)

mobilenet_model = create_mobilenetv3_model(input_shape, num_classes)
efficientnet_model = create_efficientnetb0_model(input_shape, num_classes)
shufflenet_model = create_shufflenetv2_substitute(input_shape, num_classes)
custom_model = custom_lightweight_cnn(input_shape, num_classes)

# Train models
compile_and_train_model(mobilenet_model, X_train, y_train)
compile_and_train_model(efficientnet_model, X_train, y_train)
compile_and_train_model(shufflenet_model, X_train, y_train)
compile_and_train_model(custom_model, X_train, y_train)

# --------------------- Step 6: Predict and Weighted Voting ---------------------
def get_predictions(model, X):
    return np.argmax(model.predict(X), axis=1)

weights = np.array([0.3, 0.3, 0.2, 0.2])  # Custom weights for ensemble

# Get predictions
preds = np.stack([
    get_predictions(mobilenet_model, X_test),
    get_predictions(efficientnet_model, X_test),
    get_predictions(shufflenet_model, X_test),
    get_predictions(custom_model, X_test)
], axis=0)

# Weighted voting
weighted_preds = np.apply_along_axis(
    lambda x: np.bincount(x, weights=weights, minlength=num_classes).argmax(), axis=0, arr=preds
)

y_true = np.argmax(y_test, axis=1)
y_pred = weighted_preds  # use y_pred for metrics

# --------------------- Step 7: Evaluation ---------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')  # Sensitivity
f1 = f1_score(y_true, y_pred, average='macro')
mcc = matthews_corrcoef(y_true, y_pred)

# Confusion Matrix and Derived Metrics
labels = np.unique(y_true)
cm = confusion_matrix(y_true, y_pred, labels=labels)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

TP = TP.astype(float)
TN = TN.astype(float)
FP = FP.astype(float)
FN = FN.astype(float)

specificity = TN / (TN + FP + 1e-10)
npv = TN / (TN + FN + 1e-10)
fpr = FP / (FP + TN + 1e-10)
fnr = FN / (FN + TP + 1e-10)

specificity_macro = np.mean(specificity)
npv_macro = np.mean(npv)
fpr_macro = np.mean(fpr)
fnr_macro = np.mean(fnr)

# Print Metrics
print("\n✅ Model Performance Metrics:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Sensitivity  : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity_macro:.4f}")
print(f"NPV          : {npv_macro:.4f}")
print(f"MCC          : {mcc:.4f}")
print(f"FPR          : {fpr_macro:.4f}")
print(f"FNR          : {fnr_macro:.4f}")