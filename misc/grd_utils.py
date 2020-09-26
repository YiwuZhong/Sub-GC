from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
import random
import time
import os
import sys

def get_grounding_material(infos_path, data, sents, sorted_subgraph_ind, att_weights, sort_ind, \
                           wd_to_lemma, lemma_det_id_dict, det_id_to_det_wd, \
                           grd_output, use_full_graph=False, grd_sGPN_consensus=True):
    '''
    Make up the material that is required by grounding evaluation protocol:
    find the object region / graph node that has maximum attention weight for each noun word
    '''
    # for simplicity, just load the graph file again
    f_img_id = data['infos'][0]['id']
    mask_path = 'data/flickr30k_graph_mask_1000_rm_duplicate/'+str(f_img_id)+'.npz'
    sg_path = 'data/flickr30k_sg_output_64/'+str(f_img_id)+'.npz'
    img_wh = np.load('data/flickr30k_img_wh.npy',allow_pickle=True,encoding='latin1').tolist()
    w, h = img_wh[f_img_id]
    bbox = np.load(sg_path,allow_pickle=True,encoding='latin1')['feat'].tolist()['boxes']  # bbox from SG detector
    boxes = bbox * max(w, h) / 592  # resize box back to image size

    # select best subgraph / sentence to evaluate
    subg_index = 0
    if grd_sGPN_consensus: # if True, select the sentence ranked by sGPN+consensus; if False, select best sentence ranked by sGPN
        model_path = infos_path.split('/')
        consensus_rerank_file = model_path[0] + '/' + model_path[1] + '/consensus_rerank_ind.npy'
        rerank_ind = np.load(consensus_rerank_file,allow_pickle=True,encoding='latin1').tolist()
        subg_index = rerank_ind[f_img_id][0]

    sent_used = sents[subg_index]
    grd_wd = sent_used.split()    
    if not use_full_graph:  # sub-graph captioning model
        # select best sub-graphs ranked by sGPN or sGPN+consensus-reranking
        best_subgraph_ind = sorted_subgraph_ind[subg_index].item() + 5 # the index in sampled sub-graph; first 5 are ground-truth sub-graph
        graph_mask = np.load(mask_path,allow_pickle=True,encoding='latin1')['feat'].tolist()['subgraph_mask_list'][best_subgraph_ind]
        obj_ind_this = graph_mask[1].nonzero()[0]  # the index in full graph
        att2_ind = torch.max(att_weights.data[sort_ind[subg_index].item()], dim=1)[1][:len(grd_wd)] # get maximum attention index for each word position
    else: # model that use full scene graph
        obj_ind_this = np.arange(36).astype('int')
        att2_ind = torch.max(att_weights.data[subg_index], dim=1)[1][:len(grd_wd)] # get maximum attention index for each word position          
 
    # sentence wd -> lemma -> whether lemma can be matched to detection class words -> if yes, get the detection class name
    tmp_result = {'clss':[], 'idx_in_sent':[], 'bbox':[]}
    for wd_j in range(len(grd_wd)):
        if grd_wd[wd_j] not in wd_to_lemma.keys(): 
            print('\n\n{} is not in wd_to_lemma\n\n'.format(grd_wd[wd_j]))
            continue
        lemma = wd_to_lemma[grd_wd[wd_j]]
        if lemma in lemma_det_id_dict: # lemma_det_dict: 478 detection classes, key is word, value is class id
            # att2_ind --> find the subgraph object --> find its position in full graph --> get box
            tmp_result['bbox'].append(boxes[obj_ind_this[att2_ind[wd_j]]].tolist()) # bounding box corresponding to maximum attention
            tmp_result['clss'].append(det_id_to_det_wd[lemma_det_id_dict[lemma]]) # detection class word
            tmp_result['idx_in_sent'].append(wd_j)
    grd_output[f_img_id].append(tmp_result)