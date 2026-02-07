import torch
import torch.nn as nn
## import variables
from models.classify import CircleModelV1
from data.processed import X_train, X_test, y_train, y_test

## device agnostic code
device = "cpu"
if torch.cuda.is_available(): device = 'cuda'
if torch.mps.is_available(): device = 'mps'

print("Device:", device)

model0 = CircleModelV1().to(device)

model1 = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,1)
).to(device)

print(model0, model1)

with torch.inference_mode():
    # Make predictions with the model
    untrained_preds = model1(X_test.to(device))
    print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
    print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
    print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
    print(f"\nFirst 10 test labels:\n{y_test[:10]}")




