import os 
import pandas as pd
import matplotlib.pyplot as plt
print("=" * 50)
print("HanSpeak - dataset loader")
print("=" * 50)

train_path = os.path.join("dataset", "raw", "sign_mnist_train.csv")
test_path = os.path.join("dataset", "raw", "sign_mnist_test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\n Dataset loaded successfully!\n")
print("Training Dataset Shape :", train_df.shape)
print("Testing Dataset Shape  :", test_df.shape)

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

# Display first 5 rows
print("\nFirst 5 rows of Training Dataset:")
print(train_df.head())

# Display column names
print("\nFirst 10 Columns:")
print(train_df.columns[:10])

# Check missing values
print("\nMissing Values in Training Dataset:")
print(train_df.isnull().sum().sum())

print("\nMissing Values in Testing Dataset:")
print(test_df.isnull().sum().sum())

# Display unique labels
print("\nUnique Labels:")
print(sorted(train_df['label'].astype(int).unique().tolist()))

# Count total classes
print("\nTotal Classes:")
print(train_df['label'].nunique())
# Display one sample image
import matplotlib.pyplot as plt

# First image label
label = train_df.iloc[0, 0]

# Convert pixel values into 28x28 image
image = train_df.iloc[0, 1:].values.reshape(28, 28)

# Display image
plt.figure(figsize=(4,4))
plt.imshow(image, cmap="gray")
plt.title(f"Hand Sign Label: {label}")
plt.axis("off")
plt.show()