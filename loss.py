import torch
import torch.nn as nn 


def loss_fnc(labels, outputs, masks = None):
    # Loss
    loss_fnc = nn.CrossEntropyLoss()

    # Compute the loss
    if masks is not None:
        loss = loss_fnc(outputs[masks], labels[masks])
    else: 
        loss = loss_fnc(outputs, labels)
    
    return loss

def norm_weight(weights):
    # Compute the norm of thetas:
    norm_theta = torch.norm(weights,dim = 1)
    
    # Normalize thetas:
    normalized_theta = nn.functional.normalize(weights, p = 2)

    return normalized_theta, norm_theta

def reg_loss(labels, outputs, weights, L1, L2, L3, ortho_const = False, masks = None):
    # Compute the loss
    loss = loss_fnc(labels, outputs, masks = masks)

    # Normalized weights
    normalized_theta, norm_theta = norm_weight(weights)

    # L1 penalty
    if L1 != 0:
        l1_pen = ((normalized_theta**(4)).sum(axis = 1)**(-1)).sum(axis = 0)
    else:
        l1_pen = 0.0
    
    # L2 penalty
    if L2 != 0:
        l2_pen = norm_theta.sum(axis = 0)
    else:
        l2_pen = 0.0
        
    # L3 penalty
    if L3 != 0:
        if ortho_const:
            l3_pen = torch.sum(torch.triu(torch.matmul(normalized_theta, normalized_theta.t()), diagonal = 1)**2)
        else:
            l3_pen = torch.sum(torch.triu(torch.matmul(normalized_theta, normalized_theta.t()), diagonal = 1))
    else:
        l3_pen = 0.0
    
    # Add penalty term to loss
    loss += L1*l1_pen + L2*l2_pen + L3*l3_pen

    return loss