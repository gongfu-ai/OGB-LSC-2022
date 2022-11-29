import os
import yaml
import pgl
import time
import copy
import numpy as np
import os.path as osp
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import time
from tqdm import tqdm
import torch

class UniMP(object):
    """Iterator"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def prepare_data(self):
        
        graph_file_list = []
        paper_edge_path = f'ogb-arxiv/paper_to_paper_symmetric_pgl_split'
        graph_file_list.append(paper_edge_path)
        t = time.perf_counter()
        if not osp.exists(paper_edge_path):
            log.info('Converting adjacency matrix...')
            edge_index = np.load('ogb-arxiv/edge_index.npy')
            edge_index = edge_index.T
            
            edges_new = np.zeros((edge_index.shape[0], 2))
            edges_new[:, 0] = edge_index[:, 1]
            edges_new[:, 1] = edge_index[:, 0]
            edge_index = np.vstack((edge_index, edges_new))
            edge_types = np.full([edge_index.shape[0], ], 0, dtype='int32')
            split_dict = torch.load('ogb-arxiv/split_dict.pt')
            num_nodes = split_dict['num_nodes']
            graph = Graph(edge_index, num_nodes=num_nodes, edge_feat={'edge_type': edge_types})
            graph.adj_dst_index
            graph.dump(paper_edge_path)
            log.info(f'Done! [{time.perf_counter() - t:.2f}s]')
        os.system("ln -s node_feat.npy ogb-arxiv/full_feat.npy")
        os.system("ln -s node_year.npy ogb-arxiv/all_feat_year.npy")
                
if __name__ == "__main__":
    root = 'dataset_path'
    print(root)
    dataset = UniMP(root)
    dataset.prepare_data()


