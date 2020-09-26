from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import obj_edge_vectors, load_word_vectors
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from .CaptionModel import CaptionModel
import models.lib.gcn_backbone as GBackbone
import models.lib.gpn as GPN

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    """
    for batch computation, pack sequences with different lenghth with explicit setting the batch size at each time step
    """
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

"""
Captioning model using image scene graph
"""
class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size 
        self.num_layers = opt.num_layers  
        self.drop_prob_lm = opt.drop_prob_lm 
        self.seq_length = opt.max_length or opt.seq_length 
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size 
        self.att_hid_size = opt.att_hid_size 
        self.use_bn = opt.use_bn 
        self.ss_prob = opt.sampling_prob 
        
        self.gpn = True if opt.use_gpn == 1 else False 
        self.embed_dim = opt.embed_dim 
        self.GCN_dim = opt.gcn_dim  
        self.noun_fuse = True if opt.noun_fuse == 1 else False  
        self.pred_emb_type = opt.pred_emb_type 
        self.GCN_layers = opt.gcn_layers 
        self.GCN_residual = opt.gcn_residual  
        self.GCN_use_bn = False if opt.gcn_bn == 0 else True   

        self.test_LSTM = False if getattr(opt, 'test_LSTM', 0) == 0 else True 
        self.topk_sampling = False if getattr(opt, 'use_topk_sampling', 0) == 0 else True
        self.topk_temp = getattr(opt, 'topk_temp', 0.6)
        self.the_k = getattr(opt, 'the_k', 3)
        self.sct = False if getattr(opt, 'sct', 0) == 0 else True # show-control-tell testing mode

        # feature fusion layer
        self.obj_v_proj = nn.Linear(self.att_feat_size, self.GCN_dim)
        object_names = np.load(opt.obj_name_path,encoding='latin1') # [0] is 'background'
        self.sg_obj_cnt = object_names.shape[0]
        if self.noun_fuse:
            embed_vecs = obj_edge_vectors(list(object_names), wv_dim=self.embed_dim)
            self.sg_obj_embed = nn.Embedding(self.sg_obj_cnt, self.embed_dim)
            self.sg_obj_embed.weight.data = embed_vecs.clone()
            self.obj_emb_proj = nn.Linear(self.embed_dim, self.GCN_dim)
            self.relu = nn.ReLU(inplace=True)
        predicate_names = np.load(opt.rel_name_path,encoding='latin1') # [0] is 'background'
        self.sg_pred_cnt = predicate_names.shape[0]
        p_embed_vecs = obj_edge_vectors(list(predicate_names), wv_dim=self.embed_dim)
        self.sg_pred_embed = nn.Embedding(predicate_names.shape[0], self.embed_dim)
        self.sg_pred_embed.weight.data = p_embed_vecs.clone()
        self.pred_emb_prj = nn.Linear(self.embed_dim, self.GCN_dim)

        # GCN backbone
        self.gcn_backbone = GBackbone.gcn_backbone(GCN_layers=self.GCN_layers, GCN_dim=self.GCN_dim, \
                                                   GCN_residual=self.GCN_residual, GCN_use_bn=self.GCN_use_bn)

        # GPN (sGPN)
        if self.gpn:
            self.gpn_layer = GPN.gpn_layer(GCN_dim=self.GCN_dim, hid_dim=self.att_hid_size, \
                                           test_LSTM=self.test_LSTM, use_nms=False if self.sct else True, \
                                           iou_thres=getattr(opt, 'gpn_nms_thres', 0.75), \
                                           max_subgraphs=getattr(opt, 'gpn_max_subg', 1), \
                                           use_sGPN_score=True if getattr(opt, 'use_gt_subg', 0) == 0 else False)
        else:
            self.read_out_proj = nn.Sequential(nn.Linear(self.GCN_dim, self.att_hid_size), nn.Linear(self.att_hid_size,self.GCN_dim*2))
            nn.init.constant_(self.read_out_proj[0].bias, 0)
            nn.init.constant_(self.read_out_proj[1].bias, 0)
        
        # projection layers in attention-based LSTM
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.fc_feat_size),
                                    nn.ReLU(),
                                    nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.GCN_dim, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)        

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, trip_pred=None, obj_dist=None, obj_box=None, rel_ind=None, \
                 pred_fmap=None, pred_dist=None, gpn_obj_ind=None, gpn_pred_ind=None, gpn_nrel_ind=None,gpn_pool_mtx=None):
        """
        Model feedforward: input scene graph features and sub-graph indices, output token probabilities
        fusion layers --> GCN backbone --> GPN (sGPN) --> attention-based LSTM
        """
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist, att_feats, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim

        # GCN backbone (will expand feats to 5 counterparts)
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)
        b = att_feats.size(0) # has expanded to 5 counterparts

        # sGPN
        if self.gpn:
            gpn_loss, subgraph_score, att_feats, fc_feats, att_masks = \
                self.gpn_layer(b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks)
        else: # no gpn module, baseline model with full scene graph
            gpn_loss = None
            subgraph_score = None

            # mean pooling, wo global img feats
            read_out = torch.mean(att_feats,1).detach()  # mean pool over full scene graph
            fc_feats = self.read_out_proj(read_out) 

            att_masks = att_masks[:,0,0]
            att_masks[:,:36].fill_(1.0).float()  
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)
        
        # Prepare the features for attention-based LSTM
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output  # output is probability after log_softmax at current time step, sized [batch, self.vocab_size+1]
        
        return outputs, gpn_loss, subgraph_score

    def _sample_sentences(self, fc_feats, att_feats, att_masks=None, trip_pred=None, obj_dist=None, obj_box=None, rel_ind=None, \
                                pred_fmap=None, pred_dist=None, gpn_obj_ind=None, gpn_pred_ind=None, gpn_nrel_ind=None,gpn_pool_mtx=None, opt={}):
        """
        Model inference / sentence decoding: generate captions with beam size > 1
        """
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist, att_feats, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim
        
        # GCN backbone
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)
        b = att_feats.size(0) # has expanded to 5 counterparts
        
        # GPN
        if self.gpn:
            gpn_loss, subgraph_score, att_feats, fc_feats, att_masks, keep_ind = \
                self.gpn_layer(b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks)
        else: # no gpn module, baseline model that use full graph
            gpn_loss = None
            att_feats = att_feats[0:1] # use one of 5 counterparts

            read_out = torch.mean(att_feats,1)  # mean pool over full scene graph
            fc_feats = self.read_out_proj(read_out) 

            att_masks = att_masks[0:1,0,0] 
            att_masks[:,:36].fill_(1.0).float()
            keep_ind = torch.arange(att_feats.size(0)).type_as(gpn_obj_ind)  
            subgraph_score = torch.arange(att_feats.size(0)).fill_(1.0).type_as(att_feats)

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)
            
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, None, None, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), subgraph_score, keep_ind 

    def _sample(self, fc_feats, att_feats, att_masks=None, trip_pred=None, obj_dist=None, obj_box=None, rel_ind=None, \
                pred_fmap=None, pred_dist=None, gpn_obj_ind=None, gpn_pred_ind=None, gpn_nrel_ind=None,gpn_pool_mtx=None, opt={}):
        """
        Model inference / sentence decoding: generate captions with beam size == 1 (disabling beam search)
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        return_att = True if opt.get('return_att', 0) == 1 else False

        if beam_size > 1:
            return self._sample_sentences(fc_feats, att_feats, att_masks, trip_pred, obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, \
                                      gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx, opt)
        
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist, att_feats, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim
        
        # GCN backbone
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)
        b = att_feats.size(0) # has expanded to 5 counterparts
        
        # GPN
        if self.gpn:
            gpn_loss, subgraph_score, att_feats, fc_feats, att_masks, keep_ind = \
                self.gpn_layer(b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks)
        else: # no gpn module, baseline model that use full graph
            gpn_loss = None
            att_feats = att_feats[0:1] # use one of 5 counterparts

            read_out = torch.mean(att_feats,1)  # mean pool over full scene graph
            fc_feats = self.read_out_proj(read_out) 

            att_masks = att_masks[0:1,0,0] 
            att_masks[:,:36].fill_(1.0).float()
            keep_ind = torch.arange(att_feats.size(0)).type_as(gpn_obj_ind)  
            subgraph_score = torch.arange(att_feats.size(0)).fill_(1.0).type_as(att_feats)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        att2_weights = []
        
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            if return_att:
                logprobs, state, att_weight = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, return_att=True)
                att2_weights.append(att_weight)
            else:
                logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break

            if self.topk_sampling:  # sample top-k word from a re-normalized probability distribution
                logprobs = F.log_softmax(logprobs / float(self.topk_temp), dim=1)
                tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                topk, indices = torch.topk(logprobs, self.the_k, dim=1)
                tmp = tmp.scatter(1, indices, topk)
                logprobs = tmp
                # sample the word index according to log probability (negative values)
                it = torch.distributions.Categorical(logits=logprobs.data).sample() # logits: log(probability) are negative values
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions 
            else:                
                if sample_max: # True (greedy decoding)
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()               

            # stop when all finished, unfinished: 0 or 1
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # early quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
      
        if return_att:
            # attention weights [b,20+1,N]
            att2_weights = torch.cat([_.unsqueeze(1) for _ in att2_weights], 1)
            return seq, seqLogprobs, subgraph_score, keep_ind, att2_weights
        else:
            return seq, seqLogprobs, subgraph_score, keep_ind 

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, sg_emb=None, p_sg_emb=None,return_att=False):
        """
        Attention-based LSTM feedforward
        """
        xt = self.embed(it) # 'it' contains a word index
        
        if return_att:
            output, state, att_weight = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks, return_att=return_att)
            logprobs = F.log_softmax(self.logit(output), dim=1)
            return logprobs, state, att_weight
        else:
            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
            logprobs = F.log_softmax(self.logit(output), dim=1)
            return logprobs, state

    def init_hidden(self, bsz):
        weight = self.logit.weight if hasattr(self.logit, "weight") else self.logit[0].weight
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks, sg_emb=None):
        """
        Project features and prepare for the inputs of attention-based LSTM
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) # pack sequences with different length
        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)
        
        return fc_feats, att_feats, p_att_feats, att_masks

    def feat_fusion(self, obj_dist, att_feats, pred_dist):
        """
        Fuse visual and word embedding features for nodes and edges
        """
        # fuse features (visual, embedding) for each node in graph
        if self.noun_fuse: # Sub-GC
            obj_emb = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, self.sg_obj_cnt)[:,1:].max(1)[1] + 1)).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)
            att_feats = self.obj_v_proj(att_feats)
            att_feats = self.relu(att_feats + obj_emb)
        else: # GCN-LSTM baseline that use full graph
            att_feats = self.obj_v_proj(att_feats)
        
        if self.pred_emb_type == 1: # hard emb, not including background
            pred_emb = self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt)[:,1:].max(1)[1] + 1)
        elif self.pred_emb_type == 2: # hard emb, including background
            pred_emb = self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt).max(1)[1])
        pred_fmap = self.pred_emb_prj(pred_emb).view(pred_dist.size(0), pred_dist.size(1), self.GCN_dim) 
        return att_feats, pred_fmap

"""
Attention-based LSTM
"""
class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.attention = Attention(opt)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) 
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) 

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None, sg_emb=None, p_sg_emb=None,return_att=False):
        """
        prev_h: h_lang output of previous language LSTM
        fc_feats: vector after pooling over K regions, one vector per image 
        xt: embedding of previous word
        att_feats: packed region features 
        p_att_feats: projected [packed region features]
        h_att, c_att: hidden state and cell state of attention LSTM
        h_lang, c_lang: hidden state and cell state of language LSTM
        """
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0])) # the 2nd arg is from previous att_lstm
        
        # attended region features
        if return_att:
            att, att_weight = self.attention(h_att, att_feats, p_att_feats, att_masks, return_att=return_att) 
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1])) # the 2nd arg is from previous lang_lstm

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang])) 
        
        if return_att:
            return output, state, att_weight
        else:
            return output, state

"""
Attention module in attention-based LSTM
"""
class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size 
        self.att_hid_size = opt.att_hid_size 
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None, return_att=False):
        """
        Input hidden state and region features, output the attended visual features
        """
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # [batch,512]
        att_h = att_h.unsqueeze(1).expand_as(att)            # [batch, K, 512]
        dot = att + att_h                                   # [batch, K, 512]
        dot = torch.tanh(dot) #F.tanh(dot)                  # [batch, K, 512]
        dot = dot.view(-1, self.att_hid_size)               # [(batch * K), 512]
        dot = self.alpha_net(dot)                           # [(batch * K), 1]
        dot = dot.view(-1, att_size)                        # [batch, K]
        
        weight = F.softmax(dot, dim=1)                             # [batch, K]
        if att_masks is not None:  # necessary since empty box proposals (att_mask) may exist
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # [batch, K, 1000]
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # [batch, 1000]

        if return_att:
            return att_res, weight
        else:
            return att_res

"""
Captioning model wrapper
"""
class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)
