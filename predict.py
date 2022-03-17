import argparse
import torch
import torch.nn as nn
from dgl.data import register_data_args
from model import regSGConv
from utils import *
import os


def main(args):
    
    if os.path.isfile(args.modelpath):
        print("=> loading model params '{}'".format(args.modelpath))
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage)
        model_checkpoint['args']['gpu'] = args.gpu
        model_args = argparse.Namespace(**model_checkpoint['args'])
        print("=> loaded model params '{}'".format(args.modelpath))
        print(f'model_args:{model_args}')
        print(f'args:{vars(args)}')
    else:
        print("=> no model params found at '{}'".format(args.modelpath))
    
    # # load dataset  
    data = data_loader(model_args)
    g = data['g']
    features = data['features']
    in_feats = data['input_dim']
    n_classes = data['n_classes']
    
    if g.ndata.__contains__('pred_mask'):
        pred_mask = g.ndata['pred_mask']
    else:
        print('No prediction mask found, use test_mask:')
        pred_mask = g.ndata['test_mask']
        
    
    model = regSGConv(in_feats,
                    n_classes,
                    L1 = model_args.L1,
                    L2 = model_args.L2,
                    L3 = model_args.L3,
                    k=model_args.k,
                    cached=True,
                    bias=model_args.bias)
    
    if args.gpu != -1:
        model.cuda()
    
    # Loading model
    model.load_state_dict(model_checkpoint['state_dict'])
    
    # Make prediction
    print('Predictions:...')
    fname = f'{model_args.dataset}_sgc+k_{model_args.k}+L1_{model_args.L1}+L2_{model_args.L2}+L3_{model_args.L3}_prediction.csv'
    preds = predict(model, g, features, pred_mask, class_prob = args.class_prob, fname = fname)
    

#args_list = '--modelpath ./checkpoints/citeseer_sgc+k_2+L1_0.0+L2_0.0+L3_0.0.pt'
#sys.argv = args_list.split()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regularized SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--modelpath", help="path to trained model")
    parser.add_argument("--class-prob", action = 'store_true', default = False,
            help="flag to save class probabilities" )
    args = parser.parse_args()
    main(args)