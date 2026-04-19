import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)



class MLP(nn.Module):
    def __init__(self, in_size, out_size, l_relu, hidden_size=64):
        super(MLP, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(l_relu),
            nn.Linear(hidden_size, out_size, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        return w


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
                          (-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)
        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5, l_relu=0.1, hidden_size=32,
                    head=10):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GATNet(dim_list[i], dim_he_list, gcn_dopout, l_relu, head)
        
        model_dict["C{:}".format(i + 1)] = MLP(dim_he_list[-1], num_class, l_relu,
                                                        hidden_size=hidden_size)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)

    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict


class GATNet(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout, l_relu, head):
        super().__init__()

        self.gat1 = GATConv(in_dim, hgcn_dim[0], heads=head, dropout=dropout)
        self.gat2 = GATConv(hgcn_dim[1] * head, hgcn_dim[2], dropout=dropout)
        self.l_r = l_relu
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = F.leaky_relu(x, self.l_r)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        x = F.leaky_relu(x, self.l_r)
        return x
