from torch import nn
from src.models.uplift_mlp import UpliftMLP

    
class UpliftResNet(UpliftMLP):

    class ResNet(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inputs):
            return nn.ReLU(self.module(inputs)) + inputs

    @staticmethod   
    def fetch_model(input_dim, output_dim, hidden_dim, num_hidden_layers = 1):
        layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(ResNet(nn.Linear(hidden_dim, hidden_dim)))                
        layers.append(nn.Linear(input_dim if num_hidden_layers == 0 else hidden_dim,
                                output_dim))
        model = nn.Sequential(*layers)
        return model             