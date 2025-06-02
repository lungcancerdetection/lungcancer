----------CNN-LSTM CLASSIFIER-----------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)
import cv2

# --------------------- Step 1: Load and Preprocess Tabular Data ---------------------
csv_file = "/content/drive/MyDrive/Colab Notebooks/Lung Cancer Detection/archive (82)/selected_features_with_labels.csv"
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

# Reshape features into pseudo-images: [samples, 224, 224, 3]
X_reshaped = np.array([cv2.resize(np.tile(x.reshape(1, -1), (224, 1)), (224, 224)) for x in X])
X_reshaped = np.stack([np.stack([img] * 3, axis=-1) for img in X_reshaped])

# One-hot encode labels
y_categorical = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# --------------------- Step 2: CNN-LSTM Model ---------------------
def create_cnn_lstm_model(input_shape, num_classes):
    model = models.Sequential()
    
    # CNN Feature Extractor
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    # Reshape to 2D for LSTM
    model.add(layers.Reshape((-1, 64)))  # Time steps, features

    # LSTM Layer
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.3))

    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# --------------------- Step 3: Compile and Train ---------------------
input_shape = (224, 224, 3)
model = create_cnn_lstm_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# --------------------- Step 4: Evaluate ---------------------
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

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

# --------------------- Step 5: Print Metrics ---------------------
print("\n✅ CNN-LSTM Model Performance Metrics:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Sensitivity  : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity_macro:.4f}")
print(f"NPV          : {npv_macro:.4f}")
print(f"MCC          : {mcc:.4f}")
print(f"FPR          : {fpr_macro:.4f}")
print(f"FNR          : {fnr_macro:.4f}")
------------GoogLeNet Classifier-----------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, multilabel_confusion_matrix
)
import pandas as pd
import numpy as np

# Load dataset
csv_file = "/content/drive/MyDrive/Colab Notebooks/archive (96)/selected_features_with_labels.csv"
df = pd.read_csv(csv_file)

# Clean column names
df.columns = df.columns.str.strip()
if 'label' not in df.columns:
    raise ValueError("Column 'label' not found in the dataset.")

# Drop non-numeric columns except label
df = df.drop(columns=[col for col in df.columns if df[col].dtype == 'object' and col != 'label'], errors="ignore")

# Features and labels
X = df.drop(columns=["label"], errors="ignore").values
y = df["label"].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Clean data
X = np.array(X, dtype=np.float32)
X = np.where(np.isinf(X), np.nan, X)
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# GoogLeNet Classifier
class GoogLeNetClassifier(nn.Module):
    def __init__(self, num_classes, input_features):
        super(GoogLeNetClassifier, self).__init__()
        self.googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        self.googlenet.transform_input = False
        self.googlenet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_input = nn.Linear(input_features, 1024)
        self.googlenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.fc_input(x)
        x = x.view(x.size(0), 1, 32, 32)
        return self.googlenet(x)

# Device setup
num_features = X.shape[1]
num_classes = len(np.unique(y))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GoogLeNetClassifier(num_classes, num_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Metrics calculation
def calculate_metrics(y_true, y_pred, num_classes):
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=range(num_classes))

    TP = cm[:, 1, 1]
    TN = cm[:, 0, 0]
    FP = cm[:, 0, 1]
    FN = cm[:, 1, 0]

    epsilon = 1e-10
    accuracy = accuracy_score(y_true, y_pred)
    precision = np.mean(np.divide(TP, TP + FP + epsilon))
    sensitivity = np.mean(np.divide(TP, TP + FN + epsilon))  # recall
    specificity = np.mean(np.divide(TN, TN + FP + epsilon))
    f_measure = np.mean(np.divide(2 * TP, 2 * TP + FP + FN + epsilon))
    npv = np.mean(np.divide(TN, TN + FN + epsilon))
    mcc_numer = (TP * TN - FP * FN)
    mcc_denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + epsilon
    mcc = np.mean(mcc_numer / mcc_denom)
    fpr = np.mean(np.divide(FP, FP + TN + epsilon))
    fnr = np.mean(np.divide(FN, FN + TP + epsilon))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F-measure": f_measure,
        "NPV": npv,
        "MCC": mcc,
        "FPR": fpr,
        "FNR": fnr
    }

# Training
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy, all_preds, all_labels

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc, y_pred, y_true = test(model, test_loader, criterion, device)

    metrics = calculate_metrics(y_true, y_pred, num_classes)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
---------------LeNet-DenseNet Classifier----------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
import pandas as pd
import numpy as np

# Load dataset
csv_file = "/content/drive/MyDrive/Colab Notebooks/archive (96)/selected_features_with_labels.csv"
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

# Handle infinite/missing values
X = np.array(X, dtype=np.float32)
X = np.where(np.isinf(X), np.nan, X)
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define Dense Block
class DenseBlock(nn.Module):
    def __init__(self, input_dim, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim + i * growth_rate, growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x_concat = torch.cat(features, dim=1)
            out = torch.relu(layer(x_concat))
            features.append(out)
        return torch.cat(features, dim=1)

# LeNet-DenseNet Hybrid Classifier
class LeNetDenseNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LeNetDenseNetClassifier, self).__init__()
        self.dense_block = DenseBlock(input_dim, growth_rate=32, num_layers=3)
        self.fc1 = nn.Linear(input_dim + 3*32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dense_block(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
num_features = X.shape[1]
num_classes = len(np.unique(y))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNetDenseNetClassifier(num_features, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f_measure = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        sensitivity = recall  # Macro average
        specificity = np.nan
        npv = np.nan
        fpr = np.nan
        fnr = np.nan

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F-measure": f_measure,
        "NPV": npv,
        "MCC": mcc,
        "FPR": fpr,
        "FNR": fnr
    }

# Training
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy, all_preds, all_labels

# Train
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc, y_pred, y_true = test(model, test_loader, criterion, device)
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if value is not np.nan else f"{key}: N/A")
----------yolov6----------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# --- After installing YOLOv6 repo, import the classification model builder ---
from yolov6_cls import build_model  # make sure yolov6 repo is in your PYTHONPATH

# --- Load & preprocess data ---
file_path = ''
data = pd.read_csv(file_path)

X = data.drop(columns=['label']).select_dtypes(include=[np.number])
y = data['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# --- Convert tabular data to image-like tensors ---
def tabular_to_vgg_input(X, img_size=224):
    n_samples, n_features = X.shape
    total_pixels = 3 * img_size * img_size

    if n_features < total_pixels:
        padding = np.zeros((n_samples, total_pixels - n_features))
        X_padded = np.hstack([X, padding])
    else:
        X_padded = X[:, :total_pixels]

    X_img = X_padded.reshape(n_samples, 3, img_size, img_size)
    return torch.tensor(X_img, dtype=torch.float32)

img_size = 224
X_train_img = tabular_to_vgg_input(X_train, img_size)
X_test_img = tabular_to_vgg_input(X_test, img_size)

# --- Dataset and DataLoader ---
batch_size = 32
train_ds = TensorDataset(X_train_img, torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(X_test_img, torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# --- Load pretrained YOLOv6 classification model and modify classifier ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(np.unique(y_encoded))

# Build YOLOv6 small classification model with pretrained weights
model = build_model('yolov6s-cls', pretrained=True)

# Replace the final fc layer to match your number of classes
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

# --- Training loop ---
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# --- Evaluation ---
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(yb.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# --- Metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
mcc = matthews_corrcoef(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

specificity_list, npv_list, fpr_list, fnr_list = [], [], [], []
for i in range(conf_matrix.shape[0]):
    tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
    fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
    fn = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
    tp = conf_matrix[i, i]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    specificity_list.append(specificity)
    npv_list.append(npv)
    fpr_list.append(fpr)
    fnr_list.append(fnr)

specificity = np.mean(specificity_list)
npv = np.mean(npv_list)
fpr = np.mean(fpr_list)
fnr = np.mean(fnr_list)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall (Sensitivity): {recall:.4f}")
print(f"✅ Specificity: {specificity:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ NPV: {npv:.4f}")
print(f"✅ MCC: {mcc:.4f}")
print(f"✅ FPR: {fpr:.4f}")
print(f"✅ FNR: {fnr:.4f}")
------------------------Graph-------------------------------
Dataset:1
import matplotlib.pyplot as plt

# Model names
models = ['Proposed', 'GoogleNet', 'YOLO V6', 'CNN-LSTM', 'LeNet-DenseNet']

# Updated performance metrics (new values you provided)
metrics = {
    "Accuracy":    [0.99048, 0.96096, 0.95243, 0.97989, 0.95319],
    "Specificity": [0.99389, 0.95986, 0.94642, 0.96777, 0.9773],
    "F-measure":   [0.99151, 0.97841, 0.94101, 0.95049, 0.96137],
    "Precision":   [0.97954, 0.97841, 0.95709, 0.96249, 0.95199],
    "Sensitivity": [0.99091, 0.97073, 0.95172, 0.96508, 0.95084],
    "MCC":         [0.99076, 0.97653, 0.9403, 0.95574, 0.96592],
    "NPV":         [0.98761, 0.97984, 0.94567, 0.96014, 0.96715],
    "FNR":         [0.0078, 0.02512, 0.03997, 0.02173, 0.02399],
    "FPR":         [0.00918, 0.03252, 0.04407, 0.03305, 0.03819]
}

# Define consistent colors for the models
colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#795548']
bar_width = 0.4

# Generate and show bar chart for each metric
for metric, values in metrics.items():
    plt.figure(figsize=(10, 7))
    bars = plt.bar(models, values, color=colors, width=bar_width)

    plt.xlabel("Models", fontsize=14, fontweight="bold")
    plt.ylabel(metric, fontsize=14, fontweight="bold")
    plt.title(f"{metric} Comparison Across Models", fontsize=16, fontweight="bold")
    plt.xticks(rotation=0, fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")

    # Apply y-axis limit for most metrics
    if metric not in ["FPR", "FNR"]:
        plt.ylim(0.92, 1.00)
    else:
        plt.ylim(0, max(values) + 0.01)

 

    # Save figure
    
    
    plt.show()
Dataset:2
import numpy as np
import matplotlib.pyplot as plt

# Updated model names
models = ['Proposed', 'GoogleNet', 'YOLO V6', 'CNN-LSTM', 'LeNet-DenseNet']

# Updated metric values from the latest table
metrics = {
    "Accuracy":    [0.98058, 0.95135, 0.94291, 0.97009, 0.94366],
    "Precision":   [0.96974, 0.96863, 0.94752, 0.95287, 0.94247],
    "Sensitivity": [0.981,   0.96102, 0.9422,  0.95543, 0.94133],
    "Specificity": [0.98395, 0.95026, 0.93695, 0.95809, 0.96753],
    "F-measure":   [0.9816,  0.96863, 0.9316,  0.94099, 0.95176],
    "NPV":         [0.97773, 0.97004, 0.93621, 0.95054, 0.95748],
    "MCC":         [0.98085, 0.96677, 0.9309,  0.94618, 0.95626],
    "FPR":         [0.02927, 0.03285, 0.04451, 0.03338, 0.03857],
    "FNR":         [0.01288, 0.02537, 0.04037, 0.02195, 0.02423]
}

# Color palette for bars
colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#f44336']
bar_width = 0.3

# Plot each metric as a bar chart
for metric, values in metrics.items():
    plt.figure(figsize=(12, 6))
    plt.bar(models, values, color=colors, width=bar_width)
    plt.xlabel("Models", fontsize=14, fontweight="bold")
    plt.ylabel(metric, fontsize=14, fontweight="bold")
    plt.title(f"Comparison of {metric} Across Models", fontsize=16, fontweight="bold")

    # Bold tick labels
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Tight layout
    plt.tight_layout()
    plt.show()
