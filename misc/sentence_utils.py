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

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am','the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def cal_bleu(best_ind, subgraph_bleu_material):
    '''
    output bleu scores for this batch of sentences
    subgraph_bleu_material: [#subgraphs, 4 (testlen,reflen,guess,correct), #image]
    '''
    totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*4, 'correct':[0]*4}
    testlen = 0
    reflen = 0
    for i in range(best_ind.shape[0]):
        this_subgraph = subgraph_bleu_material[best_ind[i]]
        testlen += this_subgraph['testlen'][i]
        reflen += this_subgraph['reflen'][i]
        for key in ['guess','correct']:
            for k in range(4):
                totalcomps[key][k] += this_subgraph[key][k][i]

    small = 1e-9; tiny = 1e-15; bleus = []; bleu = 1.
    for k in range(4):
        bleu *= float(totalcomps['correct'][k] + tiny) \
                / (totalcomps['guess'][k] + small)
        bleus.append(bleu ** (1./(k+1)))
    ratio = (testlen + tiny) / (reflen + small) ## N.B.: avoid zero division
    if ratio < 1:
        for k in range(4):
            bleus[k] *= math.exp(1 - 1/ratio)
    return bleus

def language_eval(dataset, align_pred, model_id, split, save_path, is_flickr=False):
    '''
    evaluate the generated sentences
    '''
    sys.path.append("misc/coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if is_flickr:
        annFile = 'misc/coco-caption/annotations/caption_flickr30k.json'
    else:
        annFile = 'misc/coco-caption/annotations/captions_val2014.json'
    coco = COCO(annFile)
    valids = coco.getImgIds()
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    
    all_scores = {}
    num_oracle = len(align_pred[0]['caption'])
    num_test_img = len(align_pred)
    all_scores['Bleu_1'] = np.zeros((num_oracle, num_test_img))
    all_scores['Bleu_2'] = np.zeros((num_oracle, num_test_img))
    all_scores['Bleu_3'] = np.zeros((num_oracle, num_test_img))
    all_scores['Bleu_4'] = np.zeros((num_oracle, num_test_img))
    all_scores['CIDEr'] = np.zeros((num_oracle, num_test_img))
    all_scores['METEOR'] = np.zeros((num_oracle, num_test_img))
    all_scores['ROUGE_L'] = np.zeros((num_oracle, num_test_img))
    all_scores['SPICE'] = np.zeros((num_oracle, num_test_img))
    all_scores['subgraph_bleu_material'] = []
    for sen_i in range(len(align_pred[0]['caption'])):  # for each sentence ind
        cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + str(sen_i) + '.json')
        preds = []
        for img_j in range(len(align_pred)): # extract the sentence in same position for all images
            entry = {'image_id': align_pred[img_j]['image_id'], 'caption': align_pred[img_j]['caption'][sen_i]}
            preds.append(entry)

        # filter results to only those in MSCOCO validation set (will be about a third)
        preds_filt = [p for p in preds if p['image_id'] in valids]
        print('using %d/%d predictions' % (len(preds_filt), len(preds)))
        json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...
        cocoRes = coco.loadRes(cache_path)
        if sen_i == 0:  # since every time cocoRes_ImgIds is the same, then only initiate cocoEval once
            cocoRes_ImgIds = cocoRes.getImgIds()
            cocoEval = COCOEvalCap(coco, cocoRes_ImgIds=cocoRes_ImgIds)
            #cocoEval.params['image_id'] = cocoRes.getImgIds()
            all_scores['image_id_list'] = list(cocoEval.gts.keys())  # fixed order of output image 

        cocoEval.evaluate(cocoRes=cocoRes)
        
        for method in cocoEval.eval_scores.keys():
            all_scores[method][sen_i,:] = np.array(cocoEval.eval_scores[method]).reshape(-1)
        all_scores['subgraph_bleu_material'].append(cocoEval.subgraph_training_bleu)

    # pick the bleu material of best subgraph in terms of individual sentence bleu score, 
    # then re-compute the score over selected sentences
    top_k = len(align_pred[0]['caption'])
    if top_k != 1:
        print('\n\nThe following is top-{}: '.format(top_k))
        bleu_dict = {'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': []}
        for metric in bleu_dict.keys():
            best_ind = np.argmax(all_scores[metric][:top_k], axis=0)
            bleu_dict[metric] = cal_bleu(best_ind, all_scores['subgraph_bleu_material'][:top_k])
        all_scores['bleu_dict'] = bleu_dict
        for b_i in range(5)[1:]:
            print('oracle {}: {}'.format('Bleu_' + str(b_i),bleu_dict['Bleu_' + str(b_i)][b_i - 1]))

        # pick maximum spice/cider/rouge/meteor score and average over images
        print('oracle spice: {}'.format(np.mean(np.max(all_scores['SPICE'][:top_k],axis=0))))
        print('oracle cider: {}'.format(np.mean(np.max(all_scores['CIDEr'][:top_k],axis=0))))
        print('oracle rouge: {}'.format(np.mean(np.max(all_scores['ROUGE_L'][:top_k],axis=0))))
        print('oracle meteor: {}'.format(np.mean(np.max(all_scores['METEOR'][:top_k],axis=0))))
        
        name = 'all_scores_{}_{}-subgraph.npy'.format(save_path[-1].split('-')[1].split('.')[0],len(align_pred[0]['caption']))
        np.save(save_path[0] + '/' + save_path[1] + '/' + name,all_scores)
        print('\n{}'.format(save_path[0] + '/' + save_path[1] + '/' + name))
