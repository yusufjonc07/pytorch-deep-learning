import torch
import torch.nn as nn
from modelv2 import LinearRegressionModelV2
from train import train_loop
from config import settings
from dataset import X_train, y_train, X_test, y_test
from plots import plot_predictions, plot_loss_curve

model_1 = LinearRegressionModelV2().to(settings.device)
torch.manual_seed(42)
print(model_1, model_1.state_dict())


# Create optimizer AFTER moving model to device
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)
loss_fn = nn.L1Loss()

train_loss_values, test_loss_values, epoch_count = train_loop(
    model_1, 
    (X_train, y_train, X_test, y_test),
    optimizer,
    loss_fn
)

# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)

## visuals
plot_predictions(predictions=y_preds.detach().cpu().numpy())
plot_loss_curve(epoch_count, train_loss_values, test_loss_values)