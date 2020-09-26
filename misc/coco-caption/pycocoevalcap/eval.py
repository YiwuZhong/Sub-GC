from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
#from .wmd.wmd import WMD
import numpy as np
import time

class COCOEvalCap:
    def __init__(self, coco, cocoRes=None, cocoRes_ImgIds=None):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        if cocoRes is not None:
            self.cocoRes = cocoRes
        if cocoRes_ImgIds is not None:
            self.params = {'image_id': cocoRes_ImgIds} # use test image id (<= ground truth image id)
        else:
            self.params = {'image_id': coco.getImgIds()} # use ground truth image id
        self.eval_scores = {}

        self.Spice = Spice()
        self.subgraph_training_bleu = None

        gts = {}
        for imgId in self.params['image_id']:
            gts[imgId] = self.coco.imgToAnns[imgId]
        tokenizer = PTBTokenizer()
        self.gts  = tokenizer.tokenize(gts)
        #np.save("imgid_list.npy",list(self.gts.keys()))

    def evaluate(self, cocoRes=None, subgraph=False):
        if cocoRes is not None:
            self.cocoRes = cocoRes
        imgIds = self.params['image_id']
        #gts = {}
        res = {}
        for imgId in imgIds:
            if not subgraph:
                #gts[imgId] = self.coco.imgToAnns[imgId]
                res[imgId] = self.cocoRes.imgToAnns[imgId]
            else:
                for sub_i in range(5):
                    gts[str(imgId)+'-'+str(sub_i)] = self.coco.imgToAnns[imgId]
                    res[str(imgId)+'-'+str(sub_i)] = [{}]
                    res[str(imgId)+'-'+str(sub_i)][0]['image_id'] = self.cocoRes.imgToAnns[imgId][0]['image_id']
                    #res[str(imgId)+'-'+str(sub_i)][0]['id'] = self.cocoRes.imgToAnns[imgId]['id']
                    res[str(imgId)+'-'+str(sub_i)][0]['caption'] = self.cocoRes.imgToAnns[imgId][0]['caption'].split("!!")[sub_i]
        
        print('tokenization...')
        tokenizer = PTBTokenizer()
        #gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        #np.save("imgid_list.npy",list(gts.keys()))  # NOTE: keys order will change after tokenize!!! 

        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (self.Spice, "SPICE"),
            #(WMD(),   "WMD"),
        ]
        
        # Compute scores
        for scorer, method in scorers:
            start = time.time()
            print('computing %s score...'%(scorer.method()))
            if type(method) == list:
                score, scores, subgraph_training_bleu = scorer.compute_score(self.gts, res)
                self.subgraph_training_bleu = subgraph_training_bleu
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m, scs)
                    self.setImgToEvalImgs(scs, list(self.gts.keys()), m)
                    print("%s: %0.3f"%(m, sc))
            elif method == "SPICE":
                score, scores, spice_scores = scorer.compute_score(self.gts, res)
                self.setEval(score, method, spice_scores)
                self.setImgToEvalImgs(scores, list(self.gts.keys()), method)
                print("%s: %0.3f"%(method, score))                
            else:
                score, scores = scorer.compute_score(self.gts, res)
                self.setEval(score, method, scores)
                self.setImgToEvalImgs(scores, list(self.gts.keys()), method)
                print("%s: %0.3f"%(method, score))
            print('\n\nUsed time {} for method {}\n\n'.format(method, time.time()-start))
        self.setEvalImgs()

    def setEval(self, score, method, scores=None):
        self.eval[method] = score
        if scores is not None:
            self.eval_scores[method] = scores
        else:
            self.eval_scores = None

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in list(self.imgToEval.items())]
