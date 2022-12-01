import os
import tqdm
import torch
import numpy as np
import torch_geometric.transforms as T
from torch import Tensor
from typing import Tuple
from torch_geometric.data import Data, Batch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
from torch_geometric_autoscale.loader import relabel_fn

'''
**ogbn-mag data format**

Data(
  num_nodes_dict={
    author=1134649,
    field_of_study=59965,
    institution=8740,
    paper=736389
  },
  edge_index_dict={
    (author, affiliated_with, institution)=[2, 1043998],
    (author, writes, paper)=[2, 7145660],
    (paper, cites, paper)=[2, 5416271],
    (paper, has_topic, field_of_study)=[2, 7505078]
  },
  x_dict={ paper=[736389, 128] },
  node_year={ paper=[736389, 1] },
  edge_reltype={
    (author, affiliated_with, institution)=[1043998, 1],
    (author, writes, paper)=[7145660, 1],
    (paper, cites, paper)=[5416271, 1],
    (paper, has_topic, field_of_study)=[7505078, 1]
  },
  y_dict={ paper=[736389, 1] }
)

'''

np.random.seed(233)

class OgbnMag(object):
    
    def __init__(self, root = '/hy-tmp/dataset'):
        self.root = root
        if os.path.exists('ogbn-mag') is False:
            os.mkdir('ogbn-mag')
        
        self.author_num=1134649
        self.field_of_study_num=59965
        self.institution_num=8740
        self.paper_num=736389
        
        self.node_num = 1134649 + 59965 + 8740 + 736389
        
        self.paper_offset = 0
        self.author_offset = self.paper_num
        self.field_of_study_offset = self.author_offset + self.author_num
        self.institution_offset = self.field_of_study_offset + self.field_of_study_num
        
        if os.path.exists('ogbn-mag/edges') is False:
            self.create_edges()
            
        self.writes = np.load('ogbn-mag/edges/writes.npy')
        self.cites = np.load('ogbn-mag/edges/cites.npy')
        self.has_topic = np.load('ogbn-mag/edges/has_topic.npy')
        self.affiliated_with = np.load('ogbn-mag/edges/affiliated_with.npy')
        
        self.graphs = []
        U, V = torch.torch.LongTensor(self.cites[0]), torch.torch.LongTensor(self.cites[1])
        adj = SparseTensor(row=U, col=V, sparse_sizes=(self.node_num, self.node_num))
        rawptr, col, _ = adj.csr()
        self.graphs.append((rawptr, col))
        
        if os.path.exists('ogbn-mag/paper_feat.npy') is False:
            dataset = PygNodePropPredDataset('ogbn-mag', f'{self.root}/OGB',pre_transform=None)
            data = dataset[0]
            self.paper_feat = data.x_dict['paper'].numpy()
            self.paper_label = data.y_dict['paper'].numpy()
            self.year = data.node_year['paper'].numpy()
            np.save('ogbn-mag/paper_feat.npy', self.paper_feat)
            np.save('ogbn-mag/paper_label.npy', self.paper_label)
            np.save('ogbn-mag/year.npy', self.year)
        else:
            self.paper_feat = np.load('ogbn-mag/paper_feat.npy')
            self.paper_label = np.load('ogbn-mag/paper_label.npy')
            self.year = np.load('ogbn-mag/year.npy')
        
        if os.path.exists('ogbn-mag/author_feat.npy') is False:
            self.author_feat = self.cal_feat(src_feat = 'paper', edges = self.writes, dst = 'author')
        else:
            self.author_feat = np.load('ogbn-mag/author_feat.npy')
            
        if os.path.exists('ogbn-mag/field_of_study_feat.npy') is False:
            self.field_of_study_feat = self.cal_feat(src_feat = 'paper', edges = self.has_topic, dst = 'field_of_study')
        else:
            self.field_of_study_feat = np.load('ogbn-mag/field_of_study_feat.npy')
        
        if os.path.exists('ogbn-mag/institution_feat.npy') is False:
            self.institution_feat = self.cal_feat(src_feat = 'author', edges = self.affiliated_with, dst = 'institution')
        else:
            self.institution_feat = np.load('ogbn-mag/institution_feat.npy')
    
        if os.path.exists('ogbn-mag/full_feat.npy') is False:
            self.full_feat = np.zeros([self.node_num, 128], dtype = np.float32)
            self.full_feat[0:self.paper_num] = self.paper_feat
            self.full_feat[self.author_offset:self.field_of_study_offset] = self.author_feat
            self.full_feat[self.field_of_study_offset:self.institution_offset] = self.field_of_study_feat
            self.full_feat[self.institution_offset:] = self.institution_feat
            np.save('ogbn-mag/full_feat.npy', self.full_feat)
        else:
            self.full_feat = np.load('ogbn-mag/full_feat.npy')
            
    def cal_feat(self, src_feat, edges, dst):
        U, V = edges[0], edges[1]
        if dst != 'author':
            U, V = V, U
        if src_feat == 'author':
            src_feat = self.author_feat
            src_offset = self.author_offset
        else:
            src_feat = self.paper_feat
            src_offset = 0
        edge_index = torch.LongTensor([U,V])
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(self.node_num, self.node_num))
        rowptr, col, _ = adj.csr()
        start = {'author':self.author_offset,
                 'field_of_study':self.field_of_study_offset,
                 'institution':self.institution_offset}[dst]
        num = {'author':self.author_num,
                 'field_of_study':self.field_of_study_num,
                 'institution':self.institution_num}[dst]
        feats = []
        step = 0
        for i in tqdm.tqdm(range(start, start + num)):
            step += 1
            x = rowptr[i]
            y = rowptr[i+1]
            if y < len(col) and col[x] < col[y]:
                feat = src_feat[col[x]-src_offset:col[y]-src_offset].mean(0)
            else:
                feat = np.zeros(128, dtype = np.float32)
            feats.append(feat)
        feats = np.array(feats)
        np.save(f'ogbn-mag/{dst}_feat.npy', feats)
        return feats
            
            
    def create_edges(self):
        dataset = PygNodePropPredDataset('ogbn-mag', f'{self.root}/OGB',pre_transform=None)
        data = dataset[0]
        affiliated_with_edge = data.edge_index_dict[('author', 'affiliated_with', 'institution')]
        writes_edge = data.edge_index_dict[('author', 'writes', 'paper')]
        cites_edge = data.edge_index_dict[('paper', 'cites', 'paper')]
        has_topic_edge = data.edge_index_dict[('paper', 'has_topic', 'field_of_study')]
        affiliated_with_edge[0] += self.author_offset
        affiliated_with_edge[1] += self.institution_offset
        writes_edge[0] += self.author_offset
        has_topic_edge[1] += self.field_of_study_offset
        
        os.mkdir('ogbn-mag/edges/')
        np.save('ogbn-mag/edges/writes.npy', writes_edge.numpy())
        np.save('ogbn-mag/edges/cites.npy', cites_edge.numpy())
        np.save('ogbn-mag/edges/has_topic.npy', has_topic_edge.numpy())
        np.save('ogbn-mag/edges/affiliated_with.npy', affiliated_with_edge.numpy())
        

def split_data():
    idx = np.array(range(0,736389))
    np.random.shuffle(idx)
    x = int(0.7 * 736389)
    y = int(0.9 * 736389)
    train_idx = idx[0:x]
    valid_idx = idx[x:y]
    test_idx = idx[y:]
    split_dict = {
        'train':train_idx,
        'valid':valid_idx,
        'test':test_idx
    }
    torch.save('ogbn-mag/split_dict.pt', split_dict)
    return split_dict

# 

        
class DataLoader(object):
    def __init__(self,  dataset, config = None, batch_size = 8,mode = 'train'):
        self.dataset = dataset
        
        self.num_features = 128
        self.num_classes = 349
        self.batch_size = batch_size
        
        if os.path.exists('ogbn-mag/split_dict.pt') is False:
            split_dict = split_data()
        else:
            split_dict = torch.load('ogbn-mag/split_dict.pt')
        self.train_idx = split_dict['train']
        self.valid_idx = split_dict['valid']
        self.test_idx = split_dict['test']
        self.mode = mode
        if mode == 'train':
            self.data_idx = self.train_idx
        elif mode == 'valid':
            self.data_idx = self.valid_idx
        else:
            self.data_idx = self.test_idx
            
        def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
            def cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

            def get_posi_angle_vec(position):
                return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

            sinusoid_table = np.array(
                [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(
                sinusoid_table[:, 1::2])  # dim 2i+1
            return sinusoid_table

        self.pos = get_sinusoid_encoding_table(200, 128)
        
        
    def sample_neighbor(self,batch_idx):
        graphs_list = []
        sample_multi_nodes = []
        batchs = []
        sample_start_nodes = []
        start_nodes = batch_idx
        re_index = {x: i for i, x in enumerate(start_nodes)}
        for i in range(2):
            graphs = []
            neigh_nodes = []
            for graph in self.dataset.graphs:
                rowptr, col = graph
                node_id = torch.LongTensor(start_nodes)
                sub_rowptr, sub_col, value, sub_n_id = relabel_fn(rowptr, col, None, node_id, False)
                adj_t = SparseTensor(rowptr=sub_rowptr.long(), col=sub_col.long(), value=value,
                             sparse_sizes=(sub_rowptr.numel() - 1, sub_n_id.numel()),
                             is_sorted=True)
                # import pdb
                # pdb.set_trace()
                graphs.append(adj_t)
                neigh_nodes.append(sub_n_id.numpy())
            
            sample_multi_nodes.append(neigh_nodes)
            graphs_list.append(graphs)
            batchs.append(len(start_nodes))
            tot_hop_set = []
            for nodes in neigh_nodes:
                one_hop_nodes = nodes[len(start_nodes):]
                tot_hop_set += list(one_hop_nodes)
            tot_hop_set = np.array(list(set(tot_hop_set)))
            for it in tot_hop_set:
                if it not in re_index:
                    re_index[it] = len(re_index)
            sample_start_nodes.append(start_nodes)
            start_nodes = np.concatenate((start_nodes, tot_hop_set))
            
        return graphs_list[::-1], sample_multi_nodes[::-1], batchs[::-1], re_index
        
    
    def __iter__(self):
        for i in range(0, len(self.data_idx), self.batch_size):
            batch_idx = self.data_idx[i:i+self.batch_size]
            graphs_list, sample_multi_nodes, batchs, re_index = self.sample_neighbor(batch_idx)
            
            y = self.dataset.paper_label[batch_idx]
        
            label_idx = list((set(re_index.keys()) - set(batch_idx)) & set(self.train_idx))
            sub_label_index = []
            for it in label_idx:
                sub_label_index.append(re_index[it])
            sub_label_index = np.array(sub_label_index)
                
            sub_label_y = self.dataset.paper_label[label_idx]
            neigh_index = [(k,v) for k,v in re_index.items()]
            neigh_index.sort(key = lambda x:x[1])
            neigh_nodes = [it[0] for it in neigh_index]
            pos = 2022 - self.dataset.year[neigh_nodes]
            pos = self.dataset.pos[pos]
            x = self.dataset.full_feat[neigh_nodes]
            x = x + pos
            yield graphs_list, sample_multi_nodes, batchs, re_index, x, y, sub_label_y, sub_label_index