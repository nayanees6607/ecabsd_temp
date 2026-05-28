import torch
from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.gcn = GCNEncoder(input_dim=23, hidden_dim=128, edge_dim=4)
        self.se3 = SE3Transformer(input_dim=128, hidden_dim=128)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        x = self.gcn(data.x, data.edge_index, edge_attr)
        x = self.se3(x)
        return x