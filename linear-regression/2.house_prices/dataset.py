import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from config import settings

batch_size = 2
shuffle = True

# Load data
df = pd.read_csv("../../data/Housing.csv")


# Encode all non-numeric columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].astype('category').cat.codes


# Split features / target
X = df.drop(columns=["price"]).values.astype('float32')
y = df["price"].values.astype('float32')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=settings.seed
)

# Normalize features (standardization)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / (X_std + 1e-8)
X_test = (X_test - X_mean) / (X_std + 1e-8)

# Normalize target
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / (y_std + 1e-8)
y_test = (y_test - y_mean) / (y_std + 1e-8)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=settings.device)
X_test  = torch.tensor(X_test,  dtype=torch.float32, device=settings.device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=settings.device)
y_test  = torch.tensor(y_test, dtype=torch.float32, device=settings.device)

# Store normalization params for inverse transform
X_NORM = {"mean": X_mean, "std": X_std}
Y_NORM = {"mean": y_mean, "std": y_std}

# Dataset + DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle
)
