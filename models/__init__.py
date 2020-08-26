from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch

from .AttModel import *

# prepare for flickr30k finetune model: restore model pretrained on COCO
def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
            #print('copy {}'.format(name))
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    word_map = np.load('data/word_mapping.npy')
    for name in ['embed.0.weight']:
        for i in range(word_map.shape[0]):
            if word_map[i] != -1:
                own_state[name][i].copy_(state_dict[name][word_map[i]])
    print("\ncopy COCO-pre-trained embedding done!\n")

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch

def setup(opt):
    # Top-down attention model
    if opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    # if finetune model on flickr30k from COCO trained model, run the following code
    #optimistic_restore(model,torch.load(os.path.join(opt.checkpoint_path, 'model-60000.pth')))

    return model
