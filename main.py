"""
This code was modified from the SGC implementation in DGL examples.
Code: https://github.com/dmlc/dgl/blob/master/examples/pytorch/sgc/sgc.py
"""

import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.data import register_data_args
from model import regSGConv
from loss import *
from utils import *

def main(args):
    # # load dataset  
    data = data_loader(args)

    g = data['g']
    features = data['features']
    labels = data['labels']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']
    in_feats = data['input_dim']
    n_classes = data['n_classes']
    n_edges = data['n_edges']

    # Checkpoint path  
    checkpoints_path = f'./checkpoints/{args.dataset}_sgc+k_{args.k}+L1_{args.L1}+L2_{args.L2}+L3_{args.L3}.pt'

    # create SGC model
    model = regSGConv(in_feats,
                    n_classes,
                    L1 = args.L1,
                    L2 = args.L2,
                    L3 = args.L3,
                    ortho = args.L3_ortho,
                    k = args.k,
                    cached = True,
                    bias = args.bias)
    
    if args.gpu != -1:
        model.cuda()
    
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)
    # use early stop
    if args.early_stop:
        metric_direction = dict(zip(args.early_stop_metric.split(','),args.early_stop_metric_dir.split(',')))
        stopper = EarlyStopping(checkpoints_path, 
                                patience=args.early_stop_patience, 
                                verbose= args.early_stop_verbose, 
                                delta = args.early_stop_delta, 
                                **metric_direction)

    # Create trainer
    train_model = make_trainer(model, reg_loss, optimizer)
    

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()

        loss = train_model(features,labels,g,train_mask)

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, reg_loss, g, features, labels, val_mask)
        
        if (epoch + 1) % args.hist_print == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Training Loss {:.4f} | Validation Loss {:.4f} | Validation Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch + 1, np.mean(dur), loss, val_loss,
                                                val_acc, n_edges / np.mean(dur) / 1000))
        
        if args.early_stop:
                val_dict = dict(zip(['loss','acc'],[val_loss,val_acc]))
                metric_value = dict([(key, val_dict[key]) for key in metric_direction.keys() if key in val_dict])
                
                if stopper(model, epoch, args, **metric_value):
                        print(f'Best model achieved at epoch: {stopper.best_epoch + 1}')
                        break
   
    print()
    if args.save_trained:
        print('Saving trained model at ./checkpoints')
        torch.save({
            'state_dict':model.state_dict(),
            'args': vars(args)
        }, checkpoints_path)
    
    if args.early_stop:
        print('loading model before testing.')
        model_checkpoint = torch.load(checkpoints_path,
                                      map_location=lambda storage, loc: storage)
        model.load_state_dict(model_checkpoint['state_dict'])
    
    _,acc = evaluate(model, reg_loss, g, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    if args.plot:
        if args.dataset in ['circle','nonlin']:
            print(f'Plotting result for synthetic data {args.dataset}!')
            syn_data_plot(features,
                        labels,
                        weights = model.fc.weight,
                        loss = loss,
                        acc = acc,
                        saveplot = True,
                        plotname=args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regularized SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.2,
            help="learning rate")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--k", type=int, default=2,
            help="number of hops")
    parser.add_argument("--L1", type=float, default=0.0,
            help="L1 constraint")
    parser.add_argument("--L2", type=float, default=0.0,
            help="L2 constraint")
    parser.add_argument("--L3", type=float, default=0.0,
            help="L3 constraint")
    parser.add_argument("--L3-ortho", action='store_true', default=False,
            help="L3 orthogonality constraint")
    parser.add_argument("--plot", action='store_true', default=False,
            help="flag to use plot for synthetic data")
    parser.add_argument("--hist-print", type = int, default=10,
            help="print training history every t epoch (default value is 10)")
    parser.add_argument("--save-trained", action='store_true', default = False,
            help="flag to save trained model")
    parser.add_argument("--early-stop", action = 'store_true', default=False,
            help="flag for early stopping")
    parser.add_argument("--early-stop-patience", type = int, default=10,
            help="patience setting for early stopping. Default is 10")
    parser.add_argument("--early-stop-metric", type = str, default = 'loss',
            help="metric used for early stopping. Default is loss")
    parser.add_argument("--early-stop-metric-dir", type = str, default = 'low',
            help="direction of metric used for early stopping [low/high]. Default is low")
    parser.add_argument("--early-stop-delta", type = float, default = 0.0,
            help="delta value used for early stopping. Default is 0.0") 
    parser.add_argument("--early-stop-verbose", action='store_true', default = False,
            help="flag to print message for early stopping")   
    args = parser.parse_args()
    print(args)

    main(args)