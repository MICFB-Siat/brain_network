import torch
import torch.nn as nn
import torch.nn.functional as F

def infoNCE_loss(q, k):
    similarity_matrix = torch.matmul(q, k.transpose(-1, -2))
    temperature = 0.07
    similarity_scores = similarity_matrix.mean(dim=1)
    logits = similarity_scores / temperature
    N = q.size(0)
    labels = torch.arange(N).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

class MLP1(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()
        self.MLP = nn.Sequential(nn.Linear(in_channels, mlp_hidden_size), nn.LeakyReLU(inplace=True), nn.Linear(mlp_hidden_size, projection_size))

    def forward(self, x):
        return self.MLP(x)

class MLP2(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()
        self.MLP = nn.Sequential(nn.Linear(in_channels, mlp_hidden_size), nn.LeakyReLU(inplace=True), nn.Linear(mlp_hidden_size, projection_size))

    def forward(self, x):
        return self.MLP(x)

class BrainFly(nn.Module):

    def __init__(self, mlp1, brain_mass, mlp2):
        super(BrainFly, self).__init__()
        self.mlp1 = mlp1
        self.brain_mass = brain_mass
        self.mlp2 = mlp2

    def forward(self, x):
        x = self.mlp1(x)
        x = self.brain_mass(x)
        x = self.mlp2(x)
        return x

class MLP_classification(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size):
        super().__init__()
        self.classification = nn.Sequential(nn.Linear(in_channels, mlp_hidden_size), nn.ReLU(), nn.Linear(mlp_hidden_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classification(x)
        x = self.sigmoid(x)
        return x
