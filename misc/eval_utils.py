from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
import random
import time
import os
import sys
import misc.utils as utils
import math
from collections import defaultdict
from misc.sentence_utils import *
from misc.grd_utils import *

random_seed = 2019
np.random.seed(random_seed)
random.seed(random_seed)

def eval_split(model, crit, loader, eval_kwargs={}, opt=None, val_model=None):
    '''
    This function contains 2 branches: 
    1. model inference: validation or testing
    2. evaluation for input sentences
    '''
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # a global configuration
    
    # grounding experiments
    return_att_weight = True if eval_kwargs.get('return_att', 0) == 1 else False
    if return_att_weight:
        assert beam_size == 1, "GVD repo only supports grounding evaluation with beam size as 1"
        gvd_all_dict = np.load('data/gvd_all_dict.npy', allow_pickle=True,encoding='latin1').tolist()
        ind_to_wd = gvd_all_dict['ind_to_wd']
        wd_to_lemma = gvd_all_dict['wd_to_lemma']
        lemma_det_id_dict = gvd_all_dict['lemma_det_id_dict']
        det_id_to_det_wd = gvd_all_dict['det_id_to_det_wd']
        grd_output = defaultdict(list)
        model_path = eval_kwargs['infos_path'].split('/')
        consensus_rerank_file = model_path[0] + '/' + model_path[1] + '/consensus_rerank_ind.npy'
        grd_sGPN_consensus = True if os.path.isfile(consensus_rerank_file) else False

    # controllability experiments
    sct_mode = True if eval_kwargs.get('sct', 0) == 1 else False

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []        
    
    # 1. run model in inference mode
    if model is not None:
        model.eval()
        loader.reset_iterator(split)
        while True:
            data = loader.get_batch(split)
            n = n + loader.batch_size
            
            if data.get('labels', None) is not None and verbose_loss: # model validation
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['trip_pred'],\
                      data['obj_dist'], data['obj_box'], data['rel_ind'], data['pred_fmap'], data['pred_dist'],\
                      data['gpn_obj_ind'], data['gpn_pred_ind'], data['gpn_nrel_ind'],data['gpn_pool_mtx']]
                tmp = [_.cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, trip_pred, obj_dist, obj_box, rel_ind, pred_fmap, pred_dist,\
                gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx = tmp

                with torch.no_grad():
                    lang_output, _, _ = model(fc_feats, att_feats, labels, att_masks, trip_pred,\
                               obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx)
                    loss = crit(lang_output, labels[:,1:], masks[:,1:]).item()
                loss_sum += loss  # only use validation loss
                loss_evals += 1
            else: # model testing
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['trip_pred'],\
                      data['obj_dist'], data['obj_box'], data['rel_ind'], data['pred_fmap'], data['pred_dist'],\
                      data['gpn_obj_ind'], data['gpn_pred_ind'], data['gpn_nrel_ind'], data['gpn_pool_mtx']]
                tmp = [_.cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, trip_pred, obj_dist, obj_box, rel_ind, pred_fmap, pred_dist,\
                gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx = tmp
                
                # send all subgraphs of a image to generate sentences
                with torch.no_grad():
                    if return_att_weight:  # grounding experiments
                        seqq, seqLogprobs, subgraph_score, keep_nms_ind, att_weights = model(fc_feats, att_feats, att_masks, trip_pred,\
                                   obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx,\
                                   opt=eval_kwargs, mode='sample')
                    else:
                        seqq, seqLogprobs, subgraph_score, keep_nms_ind = model(fc_feats, att_feats, att_masks, trip_pred,\
                                   obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx,\
                                   opt=eval_kwargs, mode='sample')
                    if not sct_mode:
                        if model.gpn: # sub-graph captioning model
                            sorted_score, sort_ind = torch.sort(subgraph_score,descending=True)
                            seq = seqq[sort_ind].data
                            subgraph_score = sorted_score.data
                            sorted_subgraph_ind = keep_nms_ind[sort_ind] # the indices are to index sub-graph in original order
                        else: # model that use full graph
                            sort_ind = torch.arange(subgraph_score.size(0)).type_as(keep_nms_ind)
                            seq = seqq.data
                            sorted_subgraph_ind = keep_nms_ind.data                               
                    else: # for show control and tell, order should be same as inputs and thus no sorting
                        valid_num = int(subgraph_score.size(0) / 2)
                        seq = seqq.data[:valid_num]
                        subgraph_score = subgraph_score.data[:valid_num]
                        sorted_subgraph_ind = keep_nms_ind[:valid_num]
                        sort_ind = keep_nms_ind[:valid_num].long()                            

                print('\nNo {}:'.format(n))
                
                if beam_size > 1 and verbose_beam:
                    keep_ind = sort_ind.cpu().numpy()
                    print('beam seach sentences of image {}:'.format(data['infos'][0]['id']))
                    for i in np.random.choice(keep_ind, size=1, replace=True):
                        print('subgraph {}'.format(i))
                        print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                        print('--' * 10)
                
                sents = utils.decode_sequence(loader.get_vocab(), seq)  # use the first beam which has highest cumulative score
                
                # save best sentence generated by all subgraphs of a image
                entry = {'image_id': data['infos'][0]['id'], 'caption': []}
                entry['subgraph_score'] = subgraph_score.cpu().numpy()
                entry['sorted_subgraph_ind'] = sorted_subgraph_ind.cpu().numpy()

                for k, sent in enumerate(sents):
                    entry['caption'].append(sent)
                predictions.append(entry)
                if verbose:
                    best_ind = torch.argmax(subgraph_score).item()
                    print('keeping {} subgraphs'.format(len(sents)))
                    print('best subgraph score sentence: \n{}'.format(entry['caption'][best_ind]))
                    print('--' * 20)
                # collect grounding material for grounding evaluation
                if return_att_weight:
                    get_grounding_material(eval_kwargs['infos_path'], data, sents, sorted_subgraph_ind, att_weights, sort_ind, \
                        wd_to_lemma, lemma_det_id_dict, det_id_to_det_wd, grd_output, \
                        use_full_graph=not model.gpn, grd_sGPN_consensus=grd_sGPN_consensus)

            if data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break

        # save prediction results
        if data.get('labels', None) is not None and verbose_loss:  # after model validation, switch back to training mode
            model.train()
            return loss_sum/loss_evals
        else:  # after model testing, save generated results
            save_path = eval_kwargs['infos_path'].split('/')

            if not sct_mode:  # sub-graph captioning
                np.save(save_path[0] + '/' + save_path[1] + '/' + 'captions_{}.npy'.format(save_path[-1].split('-')[1].split('.')[0]),predictions)
            else:  # sct mode, controllability experiments
                np.save(save_path[0] + '/' + save_path[1] + '/' + 'ctl_captions_{}.npy'.format(save_path[-1].split('-')[1].split('.')[0]),predictions)

            if return_att_weight:  # grounding experiments
                with open(save_path[0] + '/' + save_path[1] + '/' + 'grounding_file.json', 'w') as f:
                    json.dump({'results':grd_output, 'eval_mode':'gen', 'external_data':{'used':True, 'details':'grounding experiment'}}, f)

    # 2. only evaluate the generated sentences
    if model is None:
        oracle_num = eval_kwargs.get('oracle_num', 1)
        sent_cnt = []
        align_pred = []
        save_path = eval_kwargs['infos_path'].split('/')
        predictions = np.load(save_path[0] + '/' + save_path[1] + '/' + 'captions_{}.npy'.format(\
            save_path[-1].split('-')[1].split('.')[0]), allow_pickle=True,encoding='latin1').tolist()
        for p_i in range(len(predictions)):
            sent_cnt.append(len(predictions[p_i]['caption']))
            entry = {'image_id': predictions[p_i]['image_id'], 'caption': predictions[p_i]['caption'][:oracle_num]} 
            if len(entry['caption']) < oracle_num: # if subgraphs aren't engough
                for p_j in range(oracle_num)[len(entry['caption']):]:
                    entry['caption'].append(predictions[p_i]['caption'][0]) # pad with first sentence
            assert len(entry['caption']) == oracle_num
            align_pred.append(entry)
        if lang_eval == 1:
            language_eval(dataset, align_pred, eval_kwargs['id'], split, save_path, \
                            is_flickr='coco' not in eval_kwargs['input_label_h5'])