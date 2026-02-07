import torch
import torch.nn as nn

from config import settings
from model import LinearRegressionModel
from train import train_loop
from dataset import X_train, y_train, X_test, y_test, X_NORM, Y_NORM
from plots import plot_predictions, plot_loss_curve

model_0 = LinearRegressionModel().to(settings.device)

# Create optimizer AFTER moving model to device
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.01, weight_decay=0.01)
loss_fn = nn.L1Loss()

train_loss_values, test_loss_values, epoch_count = train_loop(
    model_0, 
    (X_train, y_train, X_test, y_test),
    optimizer,
    loss_fn
)
# Save the model
torch.save(model_0.state_dict(), "linear0.pth")

# # Turn model into evaluation mode
model_0.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_0(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from dataset import X_NORM, Y_NORM

# Un-normalize function for targets
inv_y = lambda y: y * Y_NORM['std'] + Y_NORM['mean']

## comparing ML regression model
linear_model = LinearRegression()
linear_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())

# Predict and un-normalize
y_preds_lr = linear_model.predict(X_test.cpu().numpy())
y_preds_lr_un = inv_y(y_preds_lr)
y_test_un = inv_y(y_test.cpu().numpy())

loss_lr = mse(y_test_un, y_preds_lr_un) ** (1/2)
print("Linear Regression model RMSE: ", loss_lr)

mean_price = y_test_un.mean()
rmse_pct = (loss_lr / mean_price) * 100
print(f"Linear Regression model RMSE as % of mean price: {rmse_pct:.2f}%")

# ## visuals
# plot_predictions(predictions=y_preds.detach().cpu().numpy())
# plot_loss_curve(epoch_count, train_loss_values, test_loss_values)

# ## load the saved model
# loaded_model = LinearRegressionModel().to(settings.device)  # Move to same device
# state_dict = torch.load("linear0.pth", weights_only=True)
# loaded_model.load_state_dict(state_dict)

# # turn on evaluation mode
# loaded_model.eval()
# with torch.inference_mode():
#     loaded_model_preds = loaded_model(X_test)

# ## check predictions are same with 
# print(torch.allclose(y_preds, loaded_model_preds))



