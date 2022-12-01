import os
import yaml
import time
import tqdm
import torch
import hydra
import models
import argparse
import numpy as np
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from dataset.ogbn_mag import OgbnMag, DataLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                        compute_micro_f1, dropout)
from tensorboardX import SummaryWriter

torch.manual_seed(123)


def mini_train(model, loader, criterion, optimizer, batch_size, device, grad_norm=None,
               edge_dropout=0.1):
    model.train()
    max_bs = 0
    total_loss = total_examples = 0
    for ii, it in tqdm.tqdm(enumerate(loader)):
        graphs_list, sample_multi_nodes, batchs, re_index, x, y, sub_label_y, sub_label_index = it
        x = torch.tensor(x).to(device)
        if x.size(0) > 15000:
            continue
        loss = 0
        max_bs = max(max_bs, x.size(0))
        sub_label_y = torch.LongTensor(sub_label_y)[:,0].to(device)
        sub_label_index = torch.LongTensor(sub_label_index ).to(device)
        y = torch.tensor(y).to(device)[:,0]
        
        for i in range(2):
            for j in range(len(graphs_list[i])):
                graphs_list[i][j] = dropout(graphs_list[i][j].to(device), p = edge_dropout)
        
        optimizer.zero_grad()
        try :
            out = model(graphs_list, sample_multi_nodes, batchs, re_index, x, sub_label_y, sub_label_index)
            loss = criterion(out, y)
            loss.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()

            total_loss += float(loss)
            total_examples += batch_size
            
        except RuntimeError as e:
            print(e)
            pass
        if ii % 20 == 0:
            print(float(loss) / batch_size)
        
    return total_loss / total_examples


@torch.no_grad()
def full_test(model, loader, device):
    model.eval()
    preds,labels = [],[]
    for ii, it in tqdm.tqdm(enumerate(loader)):
        graphs_list, sample_multi_nodes, batchs, re_index, x, y, sub_label_y, sub_label_index = it
        x = torch.tensor(x).to(device)
        if x.size(0) > 15000:
            continue
        sub_label_y = torch.LongTensor(sub_label_y)[:,0].to(device)
        sub_label_index = torch.LongTensor(sub_label_index ).to(device)
        y = torch.tensor(y).to(device)[:,0]
        
        for i in range(2):
            for j in range(len(graphs_list[i])):
                graphs_list[i][j] = graphs_list[i][j].to(device)
        try:
            out = model(graphs_list, sample_multi_nodes, batchs, re_index, x, sub_label_y, sub_label_index)
            _,pred = out.max(dim=1)
            preds.append(pred)
            labels.append(y)
        except RuntimeError as e:
            pass
    preds, labels = torch.cat(preds).cpu(), torch.cat(labels).cpu()
    return compute_micro_f1(preds, labels)
    
@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    return model(loader=loader)

def main(conf):
    
    dataset = OgbnMag()       
    train_loader = DataLoader(dataset=dataset, batch_size = config.batch_size)
    eval_loader = DataLoader(dataset=dataset, batch_size = config.batch_size, mode='test')
    
    grad_norm = None
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()

    model = getattr(models, config.model.name).GNNModel(input_size = config.model.input_size,num_class = config.model.num_class,hidden_size=config.model.hidden_size,edge_type=config.model.edge_type,alpha=config.model.alpha).to(device)
    
    def fn(step):
        warmup_steps = 2
        if step < warmup_steps:
            return 0.1
        elif step == warmup_steps:
            return 0.5
        else:
            pos_step = step - warmup_steps 
            return 0.5 * 0.75 ** (pos_step // 5)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay = config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, fn, last_epoch=-1)

    best_val_acc = test_acc = 0
    
    swriter = SummaryWriter(os.path.join(config.model_output_path, 'log'))
    
    for epoch in range(0, config.epochs):
        loss = mini_train(model, train_loader, criterion, opt, config.batch_size, device, grad_norm)
        val_acc = full_test(model, eval_loader, device)
        
        swriter.add_scalar('loss', loss, epoch)
        swriter.add_scalar('val', val_acc , epoch)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),os.path.join(config.model_output_path, str(epoch)+'.pth'))
        
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, ',
                f'Test: {val_acc:.4f}')

    print('=========================')
    print(f'Val: {best_val_acc:.4f}')


if __name__ == "__main__":
                  
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.samples = [int(i) for i in config.samples.split('-')]

    # train(config, args.ensemble_setting, args.do_eval)
    main(config)
