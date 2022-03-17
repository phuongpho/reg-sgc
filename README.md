# Regularized Simple Graph Convolution

by [Patrick Pho (Phuong Pho)](https://scholar.google.com/citations?user=yuvA4AkAAAAJ&hl=en) and [Alexander V. Mantzaris](https://scholar.google.com/citations?user=8zP4vSQAAAAJ&hl=en)

This repo is an official implementation of Regularized Simple Graph Convolution (SGC) in our [paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00366-x) - "Regularized Simple Graph Convolution (SGC) for improved interpretability of large datasets".

We incorporate a flexible regularization scheme into SGC rendering shrinkage upon three different aspects of the model's weight vectors. The $L_1$ term reduces the number of components of the weight vectors, the $L_2$ term controls the overall size of the weight vectors, and the $L_3$ terms penalizes the overlapping between the weight vectors. The proposed framework produces sparser set of fitted weights offering more meaningful interpretations.

## How to cite
If you find this repo useful, please cite:

```
@article{pho2020regularized,
  title={Regularized Simple Graph Convolution (SGC) for improved interpretability of large datasets},
  author={Pho, Phuong and Mantzaris, Alexander V},
  journal={Journal of Big Data},
  volume={7},
  number={1},
  pages={1--17},
  year={2020},
  publisher={Springer}
}
```

## Prerequisites
The dependencies can be install via:
```
pip install -r requirement.txt
```  

For GPU machine, please refer to official instruction to install suitable version of `pytorch` and `dgl`:
- [PyTorch](https://pytorch.org/)
- [Deep Graph Library - DGL](https://www.dgl.ai/pages/start.html)

## Data
Two synthetic datasets discussed in our paper can be found in `data/`. 

## Usage
### Train model
An example of incorporating $L_1 = 0.5, L_2 = 1.0,$ and $L_3 = 2.0$ into SGC fitted on Cora dataset is:
```
python main.py --dataset cora --L1 0.5 --L2 1 --L3 2
```

Use `--ortho` to impose orthogonality constraint between the weight vectors with $L_3$ term:
```
python main.py --dataset cora --L1 0.5 --L2 1 --L3 2 --L3-ortho
```

Use `--save-trained` to save trained model for inference. The trained model is save in `./checkpoints`
```
python main.py --dataset cora --L1 0.5 --L2 1 --L3 2 --L3-ortho --save-trained
```

Other useful options for training:
- `--early-stop`: turn on early stopping to reduce overfitting. Default metric is loss
- `--hist-print`: print training history at every *t* epoch
- `--plot`: plot option to use with synthetic datasets

### Prediction
We provide `predict.py` for users to make prediction on custom dataset. Before running it, you will need:
- Import your dataset as `.dgl ` in `./data`. Note that you need to include `pred_mask` masking boolean tensor in order to make prediction for unlabeled nodes. Without `pred_mask`, it will make prediction on `test_mask` nodes.
- Train model on your custom-dataset and save it. The model's name is pre-defined as `{dataset}+\_sgc+k\_{k_value}+L1_{L1_value}+L2_{L2_value}+L3_{L3_value}.pt' 

Then, user can obtain the prediction for unlabeled nodes by running:
```
python predict.py --modelpath ./checkpoints/model_name.pt
```

