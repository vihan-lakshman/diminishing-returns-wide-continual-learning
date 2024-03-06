import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_labels, width, num_layers=1):
        super().__init__()
        operations = [nn.Flatten(), nn.Linear(input_dim, width), nn.ReLU()]
        
        if num_layers > 1:
            for _ in range(num_layers-1):
                operations += [nn.Linear(width, width), nn.ReLU()]

        operations += [nn.Linear(width, num_labels)]
       
        self.model = nn.Sequential(*operations)

    def forward(self, x):
        return self.model(x)
