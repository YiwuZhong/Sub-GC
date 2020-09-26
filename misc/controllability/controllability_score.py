# Code adapted from https://github.com/aimagelab/show-control-and-tell
# speaksee package only supports Python 3.5+
# pip install speaksee==0.0.1 munkres==1.0.12

from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer
from noun_iou import NounIoU
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='ctl_captions_16000.npy',
                    help='the input file that contains the generated captions')
    args = parser.parse_args()
    
    # load the generated captions
    input_file = args.input_file
    sen_list = np.load(input_file, allow_pickle=True, encoding="latin1").tolist()
    sen_dict = {} 
    for item in sen_list:
        sen_dict[str(item['image_id'])] = item['caption']  # the generated sentences are in the order of GT captions that is grouped
    
    # load the GT materials
    noun_iou = NounIoU(pre_comp_file='flickr_noun_glove.pkl')
    order_list = np.load('order_list.npy',allow_pickle=True,encoding='latin1').tolist()
    gt_captions = np.load('sct_gt_captions.npy',allow_pickle=True,encoding='latin1').tolist()
    
    # re-order the sentence as the order of GT captions
    order_sent = [] 
    for img_id in order_list:
        order_sent.extend(sen_dict[img_id])
    print("totally {} images in the test set".format(len(sen_dict)))

    print("Computing set contrallabity results.")
    gen = {}
    gts = {}
    scores_iou = []
    for i, cap in enumerate(order_sent):
        # generated sentence string
        pred_cap = cap  # string

        gts[i] = gt_captions[i] # gt_captions[i] is a list representing a group of captions
        gen[i] = [pred_cap] # pred_cap becomes a string

        score_iou = 0.
        for c in gt_captions[i]:
            score = noun_iou.score(c, pred_cap)
            score_iou += score

        scores_iou.append(score_iou / len(gt_captions[i]))

    gts_t = PTBTokenizer.tokenize(gts)
    gen_t = PTBTokenizer.tokenize(gen)

    val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
    method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, score in zip(method, val_bleu):
        print(metric, score)

    val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
    print('METEOR', val_meteor)

    val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
    print('ROUGE_L', val_rouge)

    val_cider, _ = Cider().compute_score(gts_t, gen_t)
    print('CIDEr', val_cider)

    val_spice, _ = Spice().compute_score(gts_t, gen_t)
    print('SPICE', val_spice)

    print('Noun IoU', np.mean(scores_iou))