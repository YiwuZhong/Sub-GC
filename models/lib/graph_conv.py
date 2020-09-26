import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv_unit import _GraphConvolutionLayer_Collect

"""
Graph convolutional network
"""
class _GraphConvolutionLayer(nn.Module):
    def __init__(self, dim=2048, dim_lr=512, use_bn=False):
        super(_GraphConvolutionLayer, self).__init__()
        self.gcn_collect = _GraphConvolutionLayer_Collect(dim, dim_lr, use_bn=use_bn)

    def forward(self, feat_obj, feat_rel, map_obj_rel):
        """
        GCN collects information from neighborhoods and updates node and edge features.
        self.gcn_collect(target_node, source_node, adj_matrix, link_type_id_num)
        adj_matrix: shaped [batch, target.size(0), source.size(0)]
        """   
        # 1. collect information and update object representations
        map_sobj_rel = map_obj_rel[:, :, :, 0]
        map_oobj_rel = map_obj_rel[:, :, :, 1]
        source_rel_sub = self.gcn_collect(feat_obj, feat_rel, map_sobj_rel, 0)
        source_rel_obj = self.gcn_collect(feat_obj, feat_rel, map_oobj_rel, 1)
        feat_obj_updated = (source_rel_sub + source_rel_obj) / 2

        # 2. collect information and update edge representations
        map_rel_sobj = torch.transpose(map_obj_rel[:, :, :, 0], 1, 2) 
        map_rel_oobj = torch.transpose(map_obj_rel[:, :, :, 1], 1, 2) 
        source_obj_sub = self.gcn_collect(feat_rel, feat_obj, map_rel_sobj, 2)
        source_obj_obj = self.gcn_collect(feat_rel, feat_obj, map_rel_oobj, 3)
        feat_rel_updated = (source_obj_sub + source_obj_obj) / 2

        return feat_obj_updated, feat_rel_updated