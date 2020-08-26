from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def extract_subgraph_feats(b,N,K,L, att_feats, gpn_obj_ind, x_pred, gpn_pred_ind):
    """
    Extract the node and edge features from full scene graph by using the sub-graph indices.
    """
    # index subgraph object and predicate features
    pos_obj_ind = gpn_obj_ind[:,0,:,:]; neg_obj_ind = gpn_obj_ind[:,1,:,:]
    obj_batch_ind = torch.arange(b).view(b,1).expand(b,N*gpn_obj_ind.size(-2)).contiguous().view(-1).type_as(gpn_obj_ind)
    pos_gpn_att = att_feats[obj_batch_ind, pos_obj_ind.contiguous().view(-1)]; neg_gpn_att = att_feats[obj_batch_ind, neg_obj_ind.contiguous().view(-1)]
    
    pos_pred_ind = gpn_pred_ind[:,0,:,:].contiguous().view(-1); neg_pred_ind = gpn_pred_ind[:,1,:,:].contiguous().view(-1)
    pred_batch_ind = torch.arange(b).view(b,1).expand(b,K*gpn_pred_ind.size(-2)).contiguous().view(-1).type_as(gpn_pred_ind)
    pos_gpn_pred = x_pred[pred_batch_ind, pos_pred_ind]; neg_gpn_pred = x_pred[pred_batch_ind, neg_pred_ind]
    
    gpn_att = torch.cat((pos_gpn_att.view(-1,N,L), neg_gpn_att.view(-1,N,L)),dim=0) # pos, neg
    gpn_pred = torch.cat((pos_gpn_pred.view(-1,K,L), neg_gpn_pred.view(-1,K,L)),dim=0) # pos, neg
    return pos_obj_ind, neg_obj_ind, gpn_att, gpn_pred

def graph_pooling(N, gpn_att, gpn_pool_mtx, att_masks):
    """
    Pooling features over nodes of input sub-graphs.
    """
    # batch-wise max pooling and mean pooling
    each_pool_mtx = torch.transpose(gpn_pool_mtx, 0, 1).contiguous().view(-1,N,N) 
    clean_feats = torch.bmm(each_pool_mtx,gpn_att)
    max_feat = torch.max(clean_feats,dim=1)[0]
    mean_feat = torch.sum(clean_feats,dim=1) / torch.transpose(att_masks,0,1).sum(-1).view(-1,1)
    read_out = torch.cat((max_feat, mean_feat),dim=-1) 

    return read_out