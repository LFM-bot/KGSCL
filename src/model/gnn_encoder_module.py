import numpy as np
import torch_geometric.nn as gnn
import torch_geometric.utils as g_utils
import torch.nn.functional as F
import torch.sparse
import torch
import torch.nn as nn
from scipy import sparse
from torch.nn import Parameter

from src.utils.utils import get_activate


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, act='relu', dropout=0., hidden_dim=None):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.act = get_activate(act)
        self.dropout = nn.Dropout(dropout)
        # self.batch_norms = nn.ModuleList()

        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        for _ in range(num_layers - 1):  # previous layers
            self.gcn_layers.append(gnn.GCNConv(input_dim, hidden_dim, cached=True))
            # self.gcn_layers.append(gnn.GatedGraphConv(input_dim, hidden_dim, cached=True))
            # self.gcn_layers.append(gnn.GAT(input_dim, hidden_channels=hidden_dim,
            # out_channels=output_dim, num_layers=1))
            # self.batch_norms.append(nn.BatchNorm1d(input_dim))

        # output layer
        self.gcn_layers.append(gnn.GCNConv(hidden_dim, output_dim, cached=True))
        # self.gcn_layers.append(gnn.GatedGraphConv(hidden_dim, output_dim, cached=True))
        # self.gcn_layers.append(gnn.GAT(input_dim, hidden_channels=hidden_dim,
        # out_channels=output_dim, num_layers=1))

    def forward(self, x, edge_index, edge_weight=None, return_all_layer=False):
        """
        Args:
            x (torch.FloatTensor): [num_nodes, embed_size]
            edge_index (torch.LongTensor): [2, edge_size]
            edge_weight (torch.Tensor): [edge_size]
            return_all_layer: if return node embeddings in each gcn layer, default: False
        Returns:
            embeddings (List[torch.FloatTensor]): node embeddings after gcn layers
        """
        embeddings = []
        input_x = x
        for layer in self.gcn_layers:
            if edge_weight is not None:
                input_x = layer(input_x, edge_index, edge_weight).to(torch.float32)
            else:
                input_x = layer(input_x, edge_index).to(torch.float32)

            input_x = F.normalize(input_x, dim=-1, p=2)
            # input_x = self.dropout(self.act(input_x))
            embeddings.append(input_x)

        if return_all_layer:
            return embeddings
        return [embeddings[-1]]


class HyperGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, act='relu', dropout=0., hidden_dim=None):
        super(HyperGCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.act = get_activate(act)
        self.dropout = nn.Dropout(dropout)

        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        for _ in range(num_layers - 1):  # previous layers
            self.gcn_layers.append(gnn.HypergraphConv(input_dim, hidden_dim))

        # output layer
        self.gcn_layers.append(gnn.HypergraphConv(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, hyperedge_index, hyperedge_weight=None, return_all_layer=False):
        """
        Args:
            x (torch.FloatTensor): [num_nodes, embed_size]
            hyperedge_index (sparse matrix): incidence matrix
            hyperedge_weight (torch.LongTensor) : [edge_size]
            return_all_layer: if return node embeddings in each gcn layer, default: False
        Returns:
            embeddings (List[torch.FloatTensor]): node embeddings after gcn layers
        """
        embeddings = []
        input_x = x
        for layer in self.gcn_layers:
            if hyperedge_weight is not None:
                input_x = layer(input_x, hyperedge_index, hyperedge_weight).to(torch.float32)
            else:
                input_x = layer(input_x, hyperedge_index).to(torch.float32)
            # input_x = self.dropout(self.act(input_x))
            input_x = F.normalize(input_x, dim=-1, p=2)
            embeddings.append(input_x)

        if return_all_layer:
            return embeddings
        return [embeddings[-1]]


class GGNN(nn.Module):
    """
    2016. GATED GRAPH SEQUENCE NEURAL NETWORKS. In Proceedings of ICLR.
    """

    def __init__(self, hidden_size, step=1):
        super(GGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNN_cell(self, A, pre_hidden):
        """
        A: adjacency matrices of session graph, including input and output matrices.
            [batch, max_node_num, max_node_num * 2]
        hidden: node initial embedding, [num_items, embed_size]
        """
        neighbor_info_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(pre_hidden)) + self.b_iah
        neighbor_info_out = torch.matmul(A[:, :, A.shape[1]:2 * A.shape[1]],
                                         self.linear_edge_out(pre_hidden)) + self.b_oah
        neighbor_info = torch.cat([neighbor_info_in, neighbor_info_out], 2)
        g_i = F.linear(neighbor_info, self.w_ih, self.b_ih)  # [batch, 3, dim]
        g_h = F.linear(pre_hidden, self.w_hh, self.b_hh)
        i_r, i_u, i_c = g_i.chunk(3, 2)  # [batch, dim]
        h_r, h_u, h_c = g_h.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_u + h_u)
        candidate = torch.tanh(i_c + reset_gate * h_c)
        new_hidden = (1 - update_gate) * pre_hidden + update_gate * candidate
        return new_hidden

    def forward(self, A, hidden):
        """
        A: adjacency matrices of session graph, including input and output matrices.
            [batch, max_node_num, max_node_num * 2]
        hidden: node initial embedding, [num_items, embed_size]
        """
        for i in range(self.step):
            hidden = self.GNN_cell(A, hidden)
        return hidden


if __name__ == '__main__':
    # model
    layers = 2
    embed_size = 16
    output_size = 16
    act = 'relu'
    gcn = GCN(embed_size, output_size, layers, act=act)

    # data
    node_embedding = nn.Embedding(5, 16).weight

    # linear test
    lin = nn.Linear(embed_size, embed_size)

    row = [2, 0, 4, 0, 2, 3]
    col = [1, 3, 2, 1, 4, 1]
    adj = sparse.coo_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(5, 5))
    in_degree = adj.sum(-1).reshape(-1, 1)
    in_degree[in_degree == 0] = 1
    adj = adj.multiply(1 / in_degree)
    edge_index, edge_weight = g_utils.from_scipy_sparse_matrix(adj)

    print(node_embedding.dtype)
    print(edge_index.dtype)
    print(edge_weight.dtype)
    # forward
    res = gcn(node_embedding, edge_index, edge_weight, return_all_layer=True)

    print(res)
