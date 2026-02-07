import torch
import torch.nn as nn

from config import settings
from model import LinearRegressionModel
from modelv2 import LinearRegressionModelV2
from train import train_loop
from dataset import X_train, y_train, X_test, y_test
from plots import plot_predictions, plot_loss_curve

model_0 = LinearRegressionModel().to(settings.device)

# Create optimizer AFTER moving model to device
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
loss_fn = nn.L1Loss()

train_loss_values, test_loss_values, epoch_count = train_loop(
    model_0, 
    (X_train, y_train, X_test, y_test),
    optimizer,
    loss_fn
)
# Save the model
torch.save(model_0.state_dict(), f="linear0.pth")

# Turn model into evaluation mode
model_0.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_0(X_test)

## visuals
plot_predictions(predictions=y_preds.detach().cpu().numpy())
plot_loss_curve(epoch_count, train_loss_values, test_loss_values)

## load the saved model
loded_model = LinearRegressionModel()
state_sict = torch.load("linear0.pth")
loded_model.load_state_dict(state_sict)

# turn on evaluation mode
loded_model.eval()
with torch.inference_mode():
    loded_model_preds = loded_model(X_test)

## check predictions are same with 
print(y_preds == loded_model_preds)



