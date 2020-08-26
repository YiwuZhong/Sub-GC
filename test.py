from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import random

import time
import os

import opts
import models

from misc import eval_utils
import argparse
import misc.utils as utils
import torch

# reproducibility
random_seed = 2019
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Input arguments and options
parser = argparse.ArgumentParser()
####### Original hyper-parameters #######
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')
# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--max_length', type=int, default=20,
                help='Maximum length during sampling')
parser.add_argument('--length_penalty', type=str, default='',
                help='wu_X or avg_X, X is the alpha')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
parser.add_argument('--block_trigrams', type=int, default=0,
                help='block repeated trigram.')
parser.add_argument('--remove_bad_endings', type=int, default=0,
                help='Remove bad endings')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1, 
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, 
                help='if we need to calculate loss.')

####### Graph captioning model hyper-parameters #######
parser.add_argument('--use_gpn', type=int, default=1, 
                help='1: use GPN module in the captioning model')
parser.add_argument('--embed_dim', type=int, default=300, 
                help='dim of word embeddings')
parser.add_argument('--gcn_dim', type=int, default=1024, 
                help='dim of the node/edge features in GCN')
parser.add_argument('--noun_fuse', type=int, default=1, 
                help='1: fuse the word embedding with visual features for noun nodes')
parser.add_argument('--pred_emb_type', type=int, default=1, 
                help='predicate embedding type')
parser.add_argument('--gcn_layers', type=int, default=2, 
                help='the layer number of GCN')
parser.add_argument('--gcn_residual', type=int, default=2,
                help='2: there is a skip connection every 2 GCN layers')
parser.add_argument('--gcn_bn', type=int, default=0, 
                help='0: not use BN in GCN layers')
parser.add_argument('--sampling_prob', type=float, default=0.0, 
                help='Schedule sampling probability')
parser.add_argument('--obj_name_path', type=str, default='data/object_names_1600-0-20.npy', 
                help='the file path for object names')
parser.add_argument('--rel_name_path', type=str, default='data/predicate_names_1600-0-20.npy', 
                help='the file path for predicate names')

# parser.add_argument('--gpn_label_thres', type=float, default=0.75, 
#                 help='the threshold of positive/negative sub-graph labels during training')
parser.add_argument('--use_MRNN_split', action='store_true',
                help='use the split of MRNN on COCO Caption dataset')
parser.add_argument('--use_gt_subg', action='store_true',
                help='use the ground-truth sub-graphs (for SCT training and testing)') 
# parser.add_argument('--gpn_batch', type=int, default=2, 
#                 help='the batch size for positive/negative sub-graphs during training')    
parser.add_argument('--obj_num', type=int, default=37, 
                help='the number of detected objects + 1 dummy object')  
parser.add_argument('--rel_num', type=int, default=65, 
                help='the number of detected relationships + 1 dummy relationship')  

parser.add_argument('--num_workers', type=int, default=6, 
            help='number of workers to use')  

####### Hyper-parameters that only belongs to evaluation #######
parser.add_argument('--test_LSTM', type=int, default=1,
                help='1: generate captions, used during evaluation (testing)')
parser.add_argument('--use_topk_sampling', type=int, default=0,
                help='1: use topk sampling during decoding each word')
parser.add_argument('--topk_temp', type=float, default=0.6,
                help='the temperature used in topk sampling')
parser.add_argument('--the_k', type=int, default=3,
                help='k top candidates are used in sampling')
parser.add_argument('--gpn_nms_thres', type=float, default=0.75, 
            help='the threshold in sub-graph NMS during testing')
parser.add_argument('--gpn_max_subg', type=int, default=1, 
            help='the maximum number of sub-graphs to be kept during testing')

# sentence evaluation
parser.add_argument('--only_sent_eval', type=int, default=0, 
                help='evaluate sentence scores: 1, only run sentence evaluation; 0, only generate sentences')
parser.add_argument('--oracle_num', type=int, default=1, 
                help='how many sentences are used to calculate the top-1 accuracy')
# grounding attention return triger
parser.add_argument('--return_att', type=int, default=0, 
                help='1: return attention weight for each time step, for grounding evaluation')
# show-control-tell mode triger
parser.add_argument('--sct', type=int, default=0, 
                help='1: use sct mode where not sorting the sub-graphs and ensure the order is same as input region sets; for controllability experiments')

opt = parser.parse_args()

if __name__ == '__main__':
    # Load infos from trained model files
    with open(opt.infos_path) as f:
        infos = utils.pickle_load(f)

    # override and collect parameters
    if len(opt.input_fc_dir) == 0:
        opt.input_fc_dir = infos['opt'].input_fc_dir
        opt.input_att_dir = infos['opt'].input_att_dir
        opt.input_box_dir = getattr(infos['opt'], 'input_box_dir', '')
        opt.input_label_h5 = infos['opt'].input_label_h5
    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = 1
    if len(opt.id) == 0:
        opt.id = infos['opt'].id
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "block_trigrams"]

    # Ensure the common vars are the same; for vars only in train, copy to the opt in eval; for vars only in eval, no overrriding
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    vocab = infos['vocab'] # ix -> word mapping

    if opt.only_sent_eval == 1:  # no model inference, only evaluate generated sentences
        model = None
    elif opt.only_sent_eval == 0:  # Setup the model for inference
        model = models.setup(opt)
        model.load_state_dict(torch.load(opt.model))
        model.cuda()
        model.eval()
        
    crit = utils.LanguageModelCriterion()

    # Create the Data Loader instance
    if opt.sct == 0: # normal mode
        from dataloaders.dataloader_test import * 
    else:  # sct mode
        from dataloaders.dataloader_test_sct import * 
    if len(opt.image_folder) == 0:
      loader = DataLoader(opt)

    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']

    eval_utils.eval_split(model, crit, loader, vars(opt))

