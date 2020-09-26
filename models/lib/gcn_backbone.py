from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import _GraphConvolutionLayer

"""
GCN backbone: integrate context information within the graph and update node and edge features
"""
class gcn_backbone(nn.Module):
    def __init__(self, GCN_layers=2, GCN_dim=1024, GCN_residual=2, GCN_use_bn=False):
        super(gcn_backbone, self).__init__()
        self.GCN_layers = GCN_layers
        self.GCN_dim = GCN_dim
        self.GCN_residual = GCN_residual
        self.GCN_use_bn = GCN_use_bn
        if self.GCN_layers != 0:
            self.make_graph_encoder()

    def make_graph_encoder(self):
        self.gcn = nn.ModuleList()
        for i in range(self.GCN_layers):
            self.gcn.append(_GraphConvolutionLayer(self.GCN_dim, use_bn=self.GCN_use_bn))

    def forward(self, b,N,K,L, att_feats, obj_dist, pred_fmap, rel_ind):
        x_obj = att_feats # region feature vectors
        x_pred = pred_fmap # predicate feature vectors
        
        if self.GCN_layers != 0:
            attend_score = x_pred.data.new(b,K).fill_(1)

            # ajacency map in GCN
            map_obj_rel = self.make_map(x_obj, x_pred, attend_score, b, N, K, rel_ind[:,:,0], rel_ind[:,:,1])
            
            # GCN feed forward
            for i, gcn_layer in enumerate(self.gcn):
                x_obj, x_pred = gcn_layer(x_obj, x_pred, map_obj_rel)
                # residual skip connection
                if (i+1) % self.GCN_residual == 0:
                    x_obj = x_obj + att_feats
                    att_feats = x_obj
                    x_pred = x_pred + pred_fmap
                    pred_fmap = x_pred            

        # repeat to 5 counterparts
        x_obj = x_obj.view(b,1,N,L).expand(b,5,N,L).contiguous().view(-1,N,L)
        x_pred = x_pred.view(b,1,K,L).expand(b,5,K,L).contiguous().view(-1,K,L)

        return x_obj, x_pred

    def make_map(self, x_obj, x_pred, attend_score, batch_size, N, K, ind_subject, ind_object):
        """
        generate GCN mapping between subject and predicate, and between object and predicate
        """
        # map between sub object and obj object
        map_sobj_rel = x_obj.data.new(batch_size, N, K).zero_()
        map_oobj_rel = x_obj.data.new(batch_size, N, K).zero_()
        for i in range(batch_size):
            map_sobj_rel[i].scatter_(0, ind_subject[i].contiguous().view(1, K), attend_score[i].contiguous().view(1,K)) # row is target, col is source
            map_oobj_rel[i].scatter_(0, ind_object[i].contiguous().view(1, K), attend_score[i].contiguous().view(1,K)) # row is target, col is source
        map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 3)  # [b, N, K, 2]
        
        return map_obj_rel
    








        



