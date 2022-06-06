import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import torch
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt

# Function to import data to dgl graph object
def import_data(filename, sp_Adj ,X,labels, train_mask, val_mask, test_mask, pred_mask = None):
    '''
    This function converts inputs data into dgl graph object.
    Args:
        filename (str): file name without extension
        sp_Adj (scipy.sparse.coo.coo_matrix): adjancency matrix in coo_matrix format
        X (numpy.ndarray): 2D numpy array of features matrix
        labels (numpy.ndarray): 1D numpy array of node labels 
        train_mask (numpy.ndarray): 1D numpy array of mask of training nodes
        val_mask (numpy.ndarray): 1D numpy array of mask of validating nodes
        test_mask (numpy.ndarray): 1D numpy array of mask of test nodes
        pred_mask (numpy.ndarray): 1D numpy array of mask of unlabeled nodes for prediction
    '''
    fl_path = './data/' + filename + '.dgl'
    # Convert to dgl graph 
    G = dgl.from_scipy(sp_Adj)
    
    # Add feature
    G.ndata['feat'] = torch.FloatTensor(X)

    # Add labels
    G.ndata['label'] = torch.tensor(labels, dtype=torch.long)

    # Add train-val-test-pred masks
    G.ndata['train_mask'] = torch.tensor(train_mask, dtype= torch.bool)
    G.ndata['test_mask'] = torch.tensor(test_mask, dtype= torch.bool)
    G.ndata['val_mask'] = torch.tensor(val_mask, dtype= torch.bool)
    
    if pred_mask is not None:
        G.ndata['pred_mask'] = torch.tensor(pred_mask, dtype= torch.bool)
    
    # Save graph
    dgl.save_graphs(fl_path,G)


def data_loader(args):
    '''
    This function loads dataset 
    '''
    print(f'Loading {args.dataset} dataset:')
    # loading dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        ''' For synthetic datasets or other datasets stored in data/.'''
        dt_name = os.listdir('./data')
        check_name = []
        
        for i,v in enumerate(dt_name):
            if args.dataset+'.dgl' == v:
                check_name.append(True)
                check_id = i
            else:
                check_name.append(False)
        
        if any(check_name):
            fl_path = './data/'+dt_name[check_id]
            data,_ = dgl.load_graphs(fl_path)
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    
    if args.gpu < 0:
        device = 'cpu'
    else:
        device = 'cuda'
        
    g = g.to(device)
    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask'].to(torch.bool)
    val_mask = g.ndata['val_mask'].to(torch.bool)
    test_mask = g.ndata['test_mask'].to(torch.bool)
    in_feats = features.shape[1]
    n_classes = torch.unique(labels).numel()
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    datadict={'g':g,
            'features':features,
            'labels':labels,
            'train_mask':train_mask,
            'val_mask':val_mask,
            'test_mask': test_mask,
            'input_dim':in_feats,
            'n_classes':n_classes,
            'n_edges':n_edges}

    return datadict

def make_trainer(model, loss_fn, optimizer):
    '''
    This function produces trainer function to use in training the model
    '''
    # Builds function that performs a step in the train loop
    def trainer(features, labels, g, mask):
        '''
        This function perform training procedure: compute logits,
        update gradient, and return loss value
        '''
        # Sets model to TRAIN mode
        model.train()
        L1 = model.L1
        L2 = model.L2
        L3 = model.L3
        ortho_const = model.ortho

        # Makes predictions
        logits = model(g, features)
              
        # Compute the loss
        loss = loss_fn(labels, 
                        logits, 
                        model.fc.weight, 
                        L1, L2, L3, 
                        ortho_const = ortho_const, 
                        masks = mask, )
        
        # Computes gradients
        loss.backward()
        
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return trainer

def evaluate(model, loss_fn, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        L1 = model.L1
        L2 = model.L2
        L3 = model.L3
        ortho_const = model.ortho

        logits = model(g, features)[mask] # only compute the evaluation set
        
        labels = labels[mask]
        
        # compute loss
        loss = loss_fn(labels, logits, model.fc.weight, L1, L2, L3, ortho_const = ortho_const)
        
        # compute acc
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return loss.item(), acc
        

def predict(model, g, features, mask = None, class_prob = False, fname = None):
    '''
    This function produces node prediction
    Args:
        model (regSGConv): trained model 
        g (DGL graph): DGL graph
        features (torch.Tensor): features tensor
        mask (torch.Tensor): boolean mask tensor. If none, it makes prediction for all nodes
        class_prob (boolean): If true, compute class probabilites
        fname (str): If provided a str of file name, save prediction as fname.csv
    '''
    model.eval()
    with torch.no_grad():
        # Compute logits
        nodes = g.nodes()
        logits = model(g, features)
        
        # Compute on the mask set if needed
        if mask is not None: 
            nodes = nodes[mask]
            logits = logits[mask] 
           
        # Compute class
        _, indices = torch.max(logits, dim=1)
        
        # Save result
        preds = torch.column_stack((nodes,indices))
        
        # Compute class prob
        if class_prob:
            probs = F.softmax(logits, dim = 1)
            preds = torch.column_stack((preds,probs))
            
        if fname is not None:
            preds = preds.cpu()
            num_classes = preds.shape[1] - 2
            fmt = ",".join(['%d']*2 + ['%.4f']*num_classes)
            header = ",".join(['Node','Pred_label'] + ['prob_' + str(c) for c in range(num_classes)])
            np.savetxt(fname, preds, fmt = fmt, header = header, comments = '')
        
        return preds

class EarlyStopping:
    def __init__(self, path, patience=10, verbose=False, delta = 0.0, **metric_direction):
        '''
        Args:
            path (str): Path to save model's checkpoint.
            patience (int): How long to wait after last time validation loss (or acc) improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss (or acc) improvement. 
                            Default: False
            delta (float): Minimum percentage change in the monitored quantity (either validation loss or acc) to qualify as an improvement.
                            Default: 0.0
            **metric_direction: Keywords are names of metrics used for early stopping. Values are direction in ['low'/'high']. Use 'low' if a small quantity of metric,
                            is desirable for training and vice versa. E.g: loss = 'low', acc = 'high'. If not provided, use loss = 'low'
        '''
        if metric_direction:
            print(f'Selected metric for early stopping: {metric_direction}')
        else:
            raise ValueError("No metric provided for early stopping")

        # unpacking keys into list of string
        self.metric_name = [*metric_direction.keys()]
        # choose comparison operator w.r.t metric direction: low -> "<"; high -> ">"
        self.metric_operator = [np.less if dir == 'low' else np.greater for dir in metric_direction.values()]
        self.patience = patience
        # assign delta sign to compute reference quantity for early stopping
        self.delta = [-delta if dir == 'low' else delta for dir in metric_direction.values()]
        self.counter = 0
        self.best_score = [None]*len(metric_direction.keys())
        self.best_epoch = None
        self.lowest_loss = None
        self.path = path
        self.verbose = verbose
        self.early_stop = False
          
    def __call__(self, model, epoch, args, **metric_value):
        '''
        Args:
            metric_value: Keywords are names of metrics used for early stopping. Values are metrics's value obtained during training.
        '''
        if metric_value:
            # Check name of metric
            if set(self.metric_name) != set(metric_value.keys()):
                raise ValueError("Metric name is not matching")
        else:
            raise ValueError("Metric value is missing")
        
        score = [metric_value[key] for key in self.metric_name if key in metric_value]
        
        # if any metric is none, return true
        is_none = any(map(lambda i: i is None,self.best_score))
        
        if is_none:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, args)
        else:
            # score condition: if any metric is getting better, save model. 
            # getting better means scr is less(greater) than best_scr*[1-(+)delta/100]
            score_check = any(map(lambda scr,best_scr, op, dlt: op(scr, best_scr*(1+dlt/100)), score, self.best_score, self.metric_operator, self.delta))
            
            if score_check:
                self.best_score = score
                self.best_epoch = epoch
                self.save_checkpoint(model, args)
            else:
                self.counter += 1
                if self.counter >= 0.8*(self.patience):
                    print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            return self.early_stop

    def save_checkpoint(self, model, args):
        '''
        Saves model when score condition is met, i.e loss decreases 
        or acc increases
        '''
        if self.verbose:
            message = f'Model saved at epoch {self.best_epoch + 1}.'
            score = self.best_score
            
            if len(self.metric_name) > 1:
                for i,nm in enumerate(self.metric_name):
                    message += f' {nm}={score[i]:.4f}'
                
                print(message)
            else:
                print(f'{message} {self.metric_name[0]}={score[0]:.4f}')
        # Save model state
        torch.save({
            'state_dict':model.state_dict(),
            'args': vars(args)
        }, self.path)

def syn_data_plot(feature,label,weights,loss = None, acc = None, saveplot = False, plotname = None):
    '''
    Plotting function to visualize the results 
    using synthetic datasets. 
    The function returns a scatter plot containing data points
    and the weight vectors corresponding with two classes
    '''
    X = feature
    y_class = label
    weights = weights.detach().numpy()

    plt.scatter(x = X[:,0], y = X[:,1], c = y_class )

    
    thetahatc0 = weights[0]
    thetahatc1 = weights[1]

    plt.arrow(0,0,thetahatc0[0],thetahatc0[1],head_width = 0.05,color = 'purple')
    plt.arrow(0,0,thetahatc1[0],thetahatc1[1],head_width = 0.05, color = 'darkkhaki')

    plt.xlabel('x1')
    plt.ylabel('x2')

    # Add title 
    if (loss is not None) and (acc is not None):
        plt.title(f'loss = {loss:3.2f} \n acc = {acc:3.2f}')
    
    plt.show()
    
    if saveplot:
        if plotname is None:
            plt.savefig('fig.png')
        else:
            plotname = plotname + '.png'
            plt.savefig(plotname)