import torch.nn as nn

# class MLP_block(nn.Module):

#     def __init__(self, in_dim, out_dim):
#         super(MLP_block, self).__init__()
#         self.activation = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
#         self.layer1 = nn.Linear(in_dim, 256)
#         self.layer2 = nn.Linear(256, 128)
#         self.layer3 = nn.Linear(128, out_dim)

#     def forward(self, x):
#         x = self.activation(self.layer1(x))
#         x = self.activation(self.layer2(x))
#         output = self.layer3(x)
#         return output

class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_block, self).__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        score = self.regressor(x)
        return score