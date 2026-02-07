import torch
import torch.nn as nn

from config import settings

def train_loop(model: nn.Module, data, optimizer, loss_fn):
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    X_train, y_train, X_test, y_test = data
    
    for epoch in range(settings.epochs):
        model.train()
        y_pred = model(X_train).squeeze()
        loss = loss_fn(y_pred, y_train.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test).squeeze()
            test_loss = loss_fn(test_pred, y_test.squeeze())

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.item())
                test_loss_values.append(test_loss.item())
                print(f"Epoch: {epoch} | MSE Train Loss: {loss.item()} | MSE Test Loss: {test_loss.item()}")
    
    return train_loss_values, test_loss_values, epoch_count