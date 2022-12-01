import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import torch.nn.init as init
import math

def linear_init(input_size, hidden_size, with_bias=True, init_type='gcn'):
    linear = nn.Linear(input_size, hidden_size)
    if init_type == 'gcn':
        init.xavier_normal_(linear.weight)
        init.constant(linear.bias, 0.0)
    else:
        fan_in = input_size
        bias_bound = 1.0 / math.sqrt(fan_in)
        init.uniform_(linear.bias, a = -bias_bound, b = bias_bound)

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        init.uniform_(linear.weight, a = -weight_bound, b = weight_bound)
    return linear

class GNNModel(nn.Module):
    """Implement of GAT
    """

    def __init__(self,
                input_size,
                num_class,
                num_layers=2,
                feat_drop=0.6,
                attn_drop=0.6,
                num_heads=8,
                hidden_size=8,
                drop=0.1,
                edge_type=5,
                activation=None,
                alpha=None,
                **kwargs):
        super(GNNModel, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop = drop
        self.edge_type = edge_type
        self.gats = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.path_attns = nn.ModuleList()
        self.path_norms = nn.ModuleList()
        self.label_embed = nn.Embedding(num_class, input_size)
        self.gpr_attn = None
        if alpha:
            k_hop = num_layers
            self.gpr_attn = alpha * (1 - alpha) ** np.arange(k_hop + 1)
            self.gpr_attn[-1] = (1 - alpha) ** k_hop
        # whether to use learnable year_pos
        
        
        for i in range(self.num_layers):
            self.path_attns.append(linear_init(self.hidden_size, 1, init_type='linear'))
            self.path_norms.append(nn.BatchNorm1d(self.hidden_size,
                                             momentum=0.9))
            self.norms.append(nn.ModuleList([nn.BatchNorm1d(self.hidden_size,
                                             momentum=0.9) for _ in range(edge_type+1)]))
            if i == 0:
                self.skips.append(linear_init(input_size, self.hidden_size, init_type='linear'))
                self.gats.append(
                    nn.ModuleList(
                    [GATConv(input_size, self.hidden_size, self.num_heads, concat=False,
                           dropout=self.feat_drop, add_self_loops=False)
                    for _ in range(edge_type)]
                    )
                )
            else:
                self.skips.append(linear_init(self.hidden_size, self.hidden_size, init_type='linear'))
                self.gats.append(
                    nn.ModuleList(
                    [GATConv(self.hidden_size, self.hidden_size, self.num_heads, concat=False,
                           dropout=self.feat_drop, add_self_loops=False)
                    for _ in range(edge_type)]
                    )
                )
                
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size,
                                             momentum=0.9),
            nn.ReLU(),
            nn.Dropout(p=self.drop),
            nn.Linear(self.hidden_size, self.num_class),
        )

        self.label_mlp = nn.Sequential(
            nn.Linear(2*input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size,
                                             momentum=0.9),
            nn.ReLU(),
            nn.Dropout(p=self.drop),
            nn.Linear(self.hidden_size, input_size),
        )

        self.dropout = nn.Dropout(p=self.drop)
        self.input_drop = nn.Dropout(p=0.3)
        
        self.gats = nn.ModuleList(self.gats)
        self.skips = nn.ModuleList(self.skips)
        self.norms = nn.ModuleList(self.norms)
        self.path_attns = nn.ModuleList(self.path_attns)
        # self.path_attns_linear = nn.ModuleList(self.path_attns_linear)
        self.path_norms = nn.ModuleList(self.path_norms)


    def forward(self, graphs_list, sample_multi_nodes, batchs, re_index, feature, label_y, label_idx):

        ## label_y: 7000, 其中15%被随机替换
        # label_embed = self.label_embed(label_y)
        ## label_embed: 7000 x 128 
        # label_embed = self.input_drop(label_embed)
        ## feature: 9000 x 128, feature_embed: 7000 x 128 
        # feature_label = feature[label_idx]
        ## feature: 9000 x 128, feature_embed: 7000 x 128 
        # label_embed = torch.concat([label_embed, feature_label], axis=1)
        # label_embed = self.label_mlp(label_embed)
        # feature[label_idx] = label_embed

        for idx, (graphs, sample_nodes, batch) in enumerate(zip(graphs_list, sample_multi_nodes, batchs)):
            temp_feat = []
            # post-smoothing
            skip_feat = feature  # 当前采样图中心点对应的特征
            skip_feat = self.skips[idx](skip_feat)
            skip_feat = self.norms[idx][0](skip_feat)
            skip_feat = F.elu(skip_feat)
            temp_feat.append(skip_feat[0:batch])
            if self.gpr_attn is not None:
                if idx == 0:
                    gpr_feature = self.gpr_attn[idx] * skip_feat
                else:
                    gpr_feature = gpr_feature[0:batchs[idx-1]] + self.gpr_attn[idx] * skip_feat
            for i in range(len(graphs)):
                adj_t = graphs[i]
                nodes = sample_nodes[i]
                nodes_idx = torch.LongTensor([re_index[it] for it in nodes])
                # 分别gat, 并且取采样图中心对应特征
                feature_temp = self.gats[idx][i]((feature[nodes_idx], feature[nodes_idx][:adj_t.size(0)]), adj_t)
                feature_temp = self.norms[idx][i + 1](feature_temp)
                feature_temp = F.elu(feature_temp)[0:batch]
                # skip_feat_copy = torch.Tensor(skip_feat)
                # skip_feat_copy[nodes_idx] = feature_temp
                temp_feat.append(feature_temp)
            # all_type_feaute fuse using att
            temp_feat = torch.stack(temp_feat, axis=1)
            temp_feat_attn = self.path_attns[idx](temp_feat)
            temp_feat_attn = F.softmax(temp_feat_attn, dim=1)
            temp_feat_attn = temp_feat_attn.permute(0, 2, 1).contiguous()
            skip_feat = torch.bmm(temp_feat_attn, temp_feat)[:, 0]
            skip_feat = self.path_norms[idx](skip_feat)
            feature = self.dropout(skip_feat)[0:batch]
            
        # gpr_feature = (gpr_feature[:next_num_nodes] + self.gpr_attn[-1] * feature ) if self.gpr_attn is not None else feature
        # output = self.mlp(gpr_feature)
        output = self.mlp(feature)
        return output