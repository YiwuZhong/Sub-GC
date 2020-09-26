# Code adapted from https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch
# Supports Python2 and Python3
# pip install stanfordcorenlp==3.9.1.1

import json
from eval_grd_flickr30k_entities import FlickrGrdEval

if __name__ == '__main__':
    attn_file = 'grounding_file.json' # input file
    grd_reference = 'flickr30k_cleaned_class.json'
    split_file = 'split_ids_flickr30k_entities.json'
    val_split = 'test'

    # offline eval
    evaluator = FlickrGrdEval(reference_file=grd_reference, submission_file=attn_file,
                          split_file=split_file, val_split=[val_split],
                          iou_thresh=0.5)

    print('\nResults Summary (generated sent):')
    print('Printing attention accuracy on generated sentences...')
    prec_all, recall_all, f1_all = evaluator.grd_eval(mode='all')
    prec_loc, recall_loc, f1_loc = evaluator.grd_eval(mode='loc')
    print('\n')