from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
Sub-graph proposal network
"""
class gpn_layer(nn.Module):
    def __init__(self, GCN_dim=1024, hid_dim=512, test_LSTM=False, use_nms=True, iou_thres=0.75, max_subgraphs=1, use_sGPN_score=True):
        super(gpn_layer, self).__init__()
        self.GCN_dim = GCN_dim
        self.test_LSTM = test_LSTM
        self.use_nms = use_nms
        self.iou_thres = iou_thres
        self.max_subgraphs = max_subgraphs # how many subgraphs are kept after NMS
        self.use_sGPN_score = use_sGPN_score  # True, use sGPN network and sGPN loss; False, use gt sub-graphs (Sup. model for SCT)  

        if self.use_sGPN_score:
            self.gpn_fc = nn.Sequential(nn.Linear(self.GCN_dim * 2, hid_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(hid_dim, 1),
                                        )
            nn.init.constant_(self.gpn_fc[0].bias, 0)
            nn.init.constant_(self.gpn_fc[3].bias, 0)

        self.gpn_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.read_out_proj = nn.Sequential(nn.Linear(self.GCN_dim*2, hid_dim),
                                           nn.Linear(hid_dim, self.GCN_dim*2))
        nn.init.constant_(self.read_out_proj[0].bias, 0)
        nn.init.constant_(self.read_out_proj[1].bias, 0)


    def forward(self,b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks):
        """
        Input full graph, output sub-graph scores, sub-graph node features, and projected sub-graph read-out features
        extract sub-graph features --> pooling --> MLP --> sGPN score for each sub-graph, and index the sub-graphs with highest scores
        """
        # index subgraph node and edge features
        pos_obj_ind, neg_obj_ind, gpn_att, gpn_pred = self.extract_subgraph_feats(b,N,K,L, att_feats, gpn_obj_ind, x_pred, gpn_pred_ind)
        gb = gpn_att.size(0) 
        
        if self.use_sGPN_score:
            # max pooling and mean pooling
            read_out = self.graph_pooling(N, gpn_att, gpn_pool_mtx, att_masks) 
            # MLP to get subgraph score
            subgraph_score = self.gpn_fc(read_out)  # pos, neg
            subgraph_score = self.sigmoid(subgraph_score) # pos, neg
            gpn_target = torch.cat((subgraph_score.new(int(gb/2),1).fill_(1), subgraph_score.new(int(gb/2),1).fill_(0)),dim=0) 
            gpn_loss = self.gpn_loss(subgraph_score, gpn_target)
        else:
            # max pooling and mean pooling
            read_out = self.graph_pooling(N, gpn_att, gpn_pool_mtx, att_masks)  
            subgraph_score = read_out.new(gb,1).fill_(1)
            gpn_loss = None

        if not self.test_LSTM: # train or validation, select a positive subgraph for each sentence
            gpn_score = subgraph_score.squeeze().view(2,b,gpn_obj_ind.size(-2))  
            gpn_score = gpn_score[0] 
            gpn_ind = gpn_score.argmax(-1) 
            
            all_subgraph_obj_ind = pos_obj_ind 
            all_subgraph_att_masks = att_masks[:,0] 
            read_out = read_out.view(2,b,gpn_obj_ind.size(-2),read_out.size(-1))
            all_read_out = read_out[0]

            batch_ind = torch.arange(b).type_as(gpn_obj_ind)
            subgraph_obj_ind = all_subgraph_obj_ind[batch_ind, gpn_ind, :].view(-1)  
            att_feats = att_feats[torch.arange(b).view(b,1).expand(b,N).contiguous().view(-1).type_as(gpn_obj_ind),subgraph_obj_ind,:].view(b,N,L)
            att_masks = all_subgraph_att_masks[batch_ind, gpn_ind, :]  
            sub_read_out = all_read_out[batch_ind, gpn_ind, :].detach()  
            fc_feats = self.read_out_proj(sub_read_out) 
            
            return gpn_loss, subgraph_score, att_feats, fc_feats, att_masks

        if self.test_LSTM: # test, use all subgraphs
            assert b == 5 
            gpn_score = subgraph_score.squeeze().view(2,b,gpn_obj_ind.size(-2)) 
            gpn_score = torch.transpose(gpn_score,0,1)[0].contiguous().view(-1)  
            sen_batch = gpn_score.size(0)

            all_subgraph_obj_ind = gpn_obj_ind[0].contiguous().view(-1,N)
            att_feats = att_feats[0][all_subgraph_obj_ind.view(-1),:].view(sen_batch,N,L)
            s_att_masks = att_masks[0].contiguous().view(-1, N)  

            read_out = read_out.view(2,b,gpn_obj_ind.size(-2),read_out.size(-1))
            all_read_out = torch.transpose(read_out,0,1)[0].contiguous().view(-1, read_out.size(-1)) 
            fc_feats = self.read_out_proj(all_read_out)  
            
            keep_ind = torch.arange(gpn_score.size(0)).type_as(gpn_score)
            if self.use_nms:
                # nms to keep the subgraphs we need
                keep_ind = self.subgraph_nms(gpn_score, all_subgraph_obj_ind, att_masks)
                gpn_score = gpn_score[keep_ind]
                att_feats = att_feats[keep_ind]
                fc_feats = fc_feats[keep_ind]
                s_att_masks = s_att_masks[keep_ind]

            return gpn_loss, gpn_score, att_feats, fc_feats, s_att_masks, keep_ind

    def subgraph_nms(self, gpn_score, all_subgraph_obj_ind, att_masks):
        '''
        Apply NMS over sub-graphs.
        Input subgraph score and subgraph object index.
        Output the indices of subgraphs which will be kept. Not that the output still use the original score order.
        '''
        sort_ind = np.argsort(gpn_score.cpu().numpy())[::-1]  # Note: use sorted score (descending order) to do nms
        masks = att_masks[0].contiguous().view(-1, all_subgraph_obj_ind.size(-1)).cpu().numpy()[sort_ind,:]
        obj_ind = all_subgraph_obj_ind.cpu().numpy()[sort_ind,:]
        assert ((obj_ind != 36).nonzero()[0] != masks.nonzero()[0]).nonzero()[0].shape[0] == 0
        assert ((obj_ind != 36).nonzero()[1] != masks.nonzero()[1]).nonzero()[0].shape[0] == 0
        sorted_keep = np.ones(sort_ind.shape[0])

        for i in range(sort_ind.shape[0]):
            if sorted_keep[i] == 0:  # this subgraph has been abandoned
                continue
            else:
                this_obj_ind = np.unique(obj_ind[i][masks[i].nonzero()[0]])
                for j in range(sort_ind.shape[0])[i+1:]:
                    other_obj_ind = np.unique(obj_ind[j][masks[j].nonzero()[0]])
                    this_iou = self.cal_node_iou(this_obj_ind, other_obj_ind)
                    if this_iou > self.iou_thres:
                        sorted_keep[j] = 0

        # map back to original score order
        keep_sort_ind = sort_ind[sorted_keep == 1]
        orig_keep = np.zeros(sort_ind.shape[0])
        orig_keep[keep_sort_ind[:self.max_subgraphs]] = 1
        orig_keep_ind = torch.from_numpy(orig_keep.nonzero()[0]).type_as(all_subgraph_obj_ind)
        
        return orig_keep_ind
    
    def cal_node_iou(self, this_obj_ind, other_obj_ind):
        """
        Input node indices of 2 subgraphs.
        Output the node iou of these 2 subgraphs.
        """
        if this_obj_ind.shape[0] == 0 or other_obj_ind.shape[0] == 0: # no noun matched
            this_obj_ind = np.arange(this_obj_ind.shape[0])
        this = set(list(this_obj_ind))
        other = set(list(other_obj_ind))
        iou = len(set.intersection(this, other)) / float(len(set.union(this, other)))
        return iou

    def extract_subgraph_feats(self, b,N,K,L, att_feats, gpn_obj_ind, x_pred, gpn_pred_ind):
        """
        Extract the node and edge features from full scene graph by using the sub-graph indices.
        """
        # index subgraph object and predicate features
        pos_obj_ind = gpn_obj_ind[:,0,:,:]
        neg_obj_ind = gpn_obj_ind[:,1,:,:]
        obj_batch_ind = torch.arange(b).view(b,1).expand(b,N*gpn_obj_ind.size(-2)).contiguous().view(-1).type_as(gpn_obj_ind)
        pos_gpn_att = att_feats[obj_batch_ind, pos_obj_ind.contiguous().view(-1)]
        neg_gpn_att = att_feats[obj_batch_ind, neg_obj_ind.contiguous().view(-1)]
        
        pos_pred_ind = gpn_pred_ind[:,0,:,:].contiguous().view(-1)
        neg_pred_ind = gpn_pred_ind[:,1,:,:].contiguous().view(-1)
        pred_batch_ind = torch.arange(b).view(b,1).expand(b,K*gpn_pred_ind.size(-2)).contiguous().view(-1).type_as(gpn_pred_ind)
        pos_gpn_pred = x_pred[pred_batch_ind, pos_pred_ind]
        neg_gpn_pred = x_pred[pred_batch_ind, neg_pred_ind]
        
        gpn_att = torch.cat((pos_gpn_att.view(-1,N,L), neg_gpn_att.view(-1,N,L)),dim=0) # pos, neg
        gpn_pred = torch.cat((pos_gpn_pred.view(-1,K,L), neg_gpn_pred.view(-1,K,L)),dim=0) # pos, neg

        return pos_obj_ind, neg_obj_ind, gpn_att, gpn_pred

    def graph_pooling(self, N, gpn_att, gpn_pool_mtx, att_masks):
        """
        Pooling features over nodes of input sub-graphs.
        """
        # batch-wise max pooling and mean pooling, by diagonal matrix
        each_pool_mtx = torch.transpose(gpn_pool_mtx, 0, 1).contiguous().view(-1,N,N) 
        clean_feats = torch.bmm(each_pool_mtx,gpn_att) 
        max_feat = torch.max(clean_feats,dim=1)[0]
        mean_feat = torch.sum(clean_feats,dim=1) / torch.transpose(att_masks,0,1).sum(-1).view(-1,1)
        read_out = torch.cat((max_feat, mean_feat),dim=-1) 

        return read_out
