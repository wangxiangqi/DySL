import torch
from torch_geometric.nn import GCNConv
#import src.constants as const

LAYERS: 10
HIDDEN_SIZE: 256
GCNSI_N_FEATURES=1


class GCNSI(torch.nn.Module):
    """
    Graph Convolutional Network.
    Based on paper: https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    """

    def __init__(self):
        super(GCNSI, self).__init__()
        self.conv_first = GCNConv(GCNSI_N_FEATURES, HIDDEN_SIZE)
        self.conv = GCNConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.classifier = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, data):
        h = self.conv_first(data.x, data.edge_index)
        h = h.relu()
        for i in range(1, LAYERS):
            h = self.conv(h, data.edge_index)
            h = h.relu()
        out = self.classifier(h)
        return out