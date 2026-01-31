import torch.nn as nn
import torch

class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor)->torch.Tensor: 
        return self.weights * x + self.bias
    
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print("Model:", model_0)
print("Parametres:", list(model_0.parameters()))
print(model_0.state_dict())