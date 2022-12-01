from typing import Tuple
import os

import numpy as np
import torch
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from ogb.nodeproppred import PygNodePropPredDataset


def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',pre_transform=None)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    if os.path.exists('ogb-arxiv/edge_index.npy') is False:
        U, V = [], []
        rowptr, col, _ = data.adj_t.csr()
        for i in range(0, len(rowptr)-1):
            j = i + 1
            if rowptr[j] - rowptr[i] > 0:
                v = col[rowptr[i]:rowptr[j]]
                for it in v:
                    U.append(i)
                    V.append(it)
        edge = np.array([U, V])
        np.save('ogb-arxiv/edge_index.npy', edge)
    print('success save edge_index')
    
    node_year = data.node_year.view(-1).numpy()
    y = data.y.view(-1).numpy()
    x = data.x.numpy()
    
    np.save('ogb-arxiv/node_year.npy', node_year)
    np.save('ogb-arxiv/node_feat.npy', x)
    np.save('ogb-arxiv/node_label.npy', y)
    
    print('success save year,feat,label')
    
    split_idx = dataset.get_idx_split()
    split_idx['train'] = split_idx['train'].numpy()
    split_idx['valid'] = split_idx['valid'].numpy()
    split_idx['test'] = split_idx['test'].numpy()
    split_idx['num_nodes'] = data.num_nodes

    torch.save(split_idx, 'ogb-arxiv/split_dict.pt')
        
    print('success save split')
    
root = '/tmp/dataset'
get_arxiv(root)
