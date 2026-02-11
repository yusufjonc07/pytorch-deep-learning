import torch
import torch.nn as nn
## import variables
from models.classify import CircleModelV1
from data.processed import X_train, X_test, y_train, y_test

epochs = 1000
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def train_loop(model: nn.Module, data, optimizer, loss_fn):
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    X_train, X_test, y_train, y_test = data
    
    for epoch in range(epochs):
        model.train()
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))


        loss = loss_fn(y_pred, y_train)

        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()  # Changed from model_0 to model
        with torch.inference_mode():

            test_logits = model(X_test).squeeze()
            test_preds = torch.round(torch.sigmoid(test_logits))

            

            test_loss = loss_fn(test_preds, y_test)
            test_acc = accuracy_fn(y_pred=test_preds, y_true=y_test)

            test_loss = loss_fn(test_preds, y_test.type(torch.float))
            
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.item())
                test_loss_values.append(test_loss.item())
                print(f"Train: {epoch}th epoch | Loss: {loss.item()} | Acc: {acc}")
                print(f"Test: {epoch}th epoch | Loss: {test_loss.item()} | Acc: {test_acc}")
    
    return train_loss_values, test_loss_values, epoch_count

## device agnostic code
device = "cpu"
if torch.cuda.is_available(): device = 'cuda'
if torch.mps.is_available(): device = 'mps'


model1 = CircleModelV1().to(device)



optimizer = torch.optim.SGD(lr=0.001, params=model1.parameters())
lossfn = nn.BCELoss()

train_loss_values, test_loss_values, epoch_count = train_loop(
    model1, (X_train, X_test, y_train, y_test), 
    optimizer, lossfn
)

print(train_loss_values, test_loss_values, epoch_count)


