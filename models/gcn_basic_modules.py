import jittor as jt
from jittor import nn
import numpy as np

def MLP(channels, batch_norm=True):
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        layers.append(nn.Relu())
        if batch_norm:
            layers.append(nn.BatchNorm1d(channels[i], momentum=0.1))
    return nn.Sequential(*layers)

class EdgeConv(jt.Module):
    def __init__(self, in_channels, out_channels, nn_model, aggr='max'):
        super().__init__()
        self.nn = nn_model
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def execute(self, x, edge_index):
        num_nodes = x.shape[0] # x: [num_nodes, in_channels]
        row, col = edge_index  # edge_index shape: [2, num_edges]

        # Message passing: x_j - x_i, concat with x_i
        x_i = x[row, :]
        x_j = x[col, :]
        m = jt.concat([x_i, x_j - x_i], dim=1)  # [num_edges, 2*in_channels]

        # Feed through neural network
        messages = self.nn(m)

        # Aggregate messages
        if self.aggr == 'max':
            output = jt.zeros((num_nodes, messages.shape[1]), dtype=messages.dtype)
            idx = row.unsqueeze(1).expand(-1, messages.shape[1])
            output = jt.scatter(output, idx, messages, reduce='maximum')
        elif self.aggr == 'mean':
            output = jt.zeros((num_nodes, messages.shape[1]), dtype=messages.dtype)
            idx = row.unsqueeze(1).expand(-1, messages.shape[1])
            output = jt.scatter(output, idx, messages, reduce='mean')
        else:
            raise ValueError("Unsupported aggregation type: {}".format(self.aggr))

        return output

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class GCU(jt.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super().__init__()
        hidden = out_channels // 2
        # one-ring mesh neighbors
        self.edge_conv_tpl = EdgeConv(in_channels, out_channels // 2, MLP([in_channels * 2, hidden, hidden]), aggr=aggr)
        # geodesic neighborhoods
        self.edge_conv_geo = EdgeConv(in_channels, out_channels // 2, MLP([in_channels * 2, hidden, hidden]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def execute(self, x, tpl_edge_index, geo_edge_index):
        x_tpl = self.edge_conv_tpl(x, tpl_edge_index)
        x_geo = self.edge_conv_geo(x, geo_edge_index)
        x_out = jt.concat([x_tpl, x_geo], dim=1)
        x_out = self.mlp(x_out)
        return x_out
