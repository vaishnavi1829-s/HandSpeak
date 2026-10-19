import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

print("=" * 50)
print("HandSpeak - Data Preprocessing")
print("=" * 50)

# Dataset paths
train_path = os.path.join("dataset", "raw", "sign_mnist_train.csv")
test_path = os.path.join("dataset", "raw", "sign_mnist_test.csv")

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\nDatasets loaded successfully!")

print("Training Dataset Shape :", train_df.shape)
print("Testing Dataset Shape  :", test_df.shape)
# =====================================
# Separate Features (X) and Labels (y)
# =====================================

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

print("\nFeatures and Labels separated successfully!")

print("\nTraining Features Shape :", X_train.shape)
print("Training Labels Shape   :", y_train.shape)

print("\nTesting Features Shape  :", X_test.shape)
print("Testing Labels Shape    :", y_test.shape)
# =====================================
# Reshape Images
# =====================================

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

print("\nImages reshaped successfully!")

print("\nTraining Image Shape :", X_train.shape)
print("Testing Image Shape  :", X_test.shape)
# =====================================
# Split Training and Validation Data
# =====================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\nTraining and Validation datasets created successfully!")

print("\nTraining Images :", X_train.shape)
print("Validation Images :", X_val.shape)

print("\nTraining Labels :", y_train.shape)
print("Validation Labels :", y_val.shape)
# =====================================
# One-Hot Encode Labels
# =====================================

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print("\nLabels encoded successfully!")

print("\nTraining Labels Shape :", y_train.shape)
print("Validation Labels Shape :", y_val.shape)
print("Testing Labels Shape :", y_test.shape)