import torch
from config import settings

# Create data
torch.manual_seed(settings.seed)
weight = 0.7
bias = 0.3


start = 0
end = 1
step = 0.02


X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Move to device

X_train = X_train.to(settings.device)
X_test = X_test.to(settings.device)
y_train = y_train.to(settings.device)
y_test = y_test.to(settings.device)
