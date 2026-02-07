import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=12, out_features=1)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.linear_layer(x)
