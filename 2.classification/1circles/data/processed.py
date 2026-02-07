import torch
from sklearn.model_selection import train_test_split
from .raw import X, y

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print("Types:", X.dtype, y.dtype, "\n", "-"*30)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("Lenghts:", len(X_train), len(X_test), len(y_train), len(y_test), "\n", "-"*30)