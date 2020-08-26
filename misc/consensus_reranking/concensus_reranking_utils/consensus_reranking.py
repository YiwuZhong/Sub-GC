'''
Main script of the consensus reranking

Authors: Junhua Mao <mjhustc@ucla.edu>
'''

import numpy as np
import os
from scipy.spatial import distance
import json
import sys
import logging

import conf_cr
import bleu_score_util as bs_util

# library for COCO evaluation
sys.path.append('./external/coco-caption')
sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.eval_pair_cider import COCOEvalCapPairCider

# logger settings
logging.basicConfig(
    format='[%(levelname)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('CR')
logger.setLevel(logging.INFO)

class Consensus_Reranking:
    '''
    Main Class for consensus reranking
    '''
    def __init__(self, input_args):
        self.conf = conf_cr.Conf_Cr(input_args)
        
        self.ref_feat = []
        self.tar_feat = []
        
        self.anno_list_ref = []
        self.anno_list_hypo = []
        
        self.NNimg_list = []
        
        self.anno_list_reranked = []
        
        self.cocoEval = []
        
    def load_anno_ref_hypo(self):
        # Load reference annotation list
        fname_anno_list_ref = self.conf.fname_anno_list_ref
        self.anno_list_ref = np.load(fname_anno_list_ref,allow_pickle=True).tolist()
        # Load hypo annotation list
        fname_hypo_list = self.conf.fname_hypo_list
        self.anno_list_hypo = np.load(fname_hypo_list,allow_pickle=True).tolist()
        logger.info('Successfully load %d reference and %d hypothese annotations' \
            % (len(self.anno_list_ref), len(self.anno_list_hypo) ) )
            
    def find_NNimg(self):
        fol_cache = os.path.join(self.conf.fol_root_cache, self.conf.name_cache)
        if not os.path.exists(fol_cache):
            os.makedirs(fol_cache)
        fname_NNimg_list = os.path.join(fol_cache, 'NNimg_list_%s_%s_top%d.npy' \
            % (self.conf.name_feat, self.conf.distance_metric, self.conf.num_NNimg) )
        if os.path.isfile(fname_NNimg_list):
            self.NNimg_list = np.load(fname_NNimg_list,allow_pickle=True).tolist()
            logger.info('NN images list is loaded')
            return
    
        num_tr = len(self.anno_list_ref)
        num_te = len(self.anno_list_hypo)
        assert(num_tr>0 and num_te>0)
        
        # Load and arrange features
        fname_img_feat_tr = os.path.join(fol_cache, '%s_arr_tr.npy' % self.conf.name_feat)
        fname_img_feat_te = os.path.join(fol_cache, '%s_arr_te.npy' % self.conf.name_feat)
        if not (os.path.isfile(fname_img_feat_tr) and os.path.isfile(fname_img_feat_te) ):
            # Load image feature dictionary
            fname_dct_feat = self.conf.fname_dct_feat
            dct_feat = np.load(fname_dct_feat,allow_pickle=True).tolist()
            logger.info('Dictionary of image features loaded from %s' \
                % fname_dct_feat )
            # Arrange the image feature with the same order of 
            # anno_list_ref and anno_list_hypo
            feat_arr_tr = np.zeros((num_tr, self.conf.dim_image_feat), dtype=np.float64)
            for (ind, anno) in enumerate(self.anno_list_ref):
                feat_arr_tr[ind, :] = dct_feat[anno['id']]
            feat_arr_te = np.zeros((num_te, self.conf.dim_image_feat), dtype=np.float64)
            for (ind, anno) in enumerate(self.anno_list_hypo):
                feat_arr_te[ind, :] = dct_feat[anno['id']]
            if self.conf.del_useless:
                del dct_feat
            np.save(fname_img_feat_tr, feat_arr_tr)
            np.save(fname_img_feat_te, feat_arr_te)
            logger.info('Image features arranged and saved')
        else:
            feat_arr_tr = np.load(fname_img_feat_tr,allow_pickle=True)
            feat_arr_te = np.load(fname_img_feat_te,allow_pickle=True)
            assert(feat_arr_tr.shape[1]==self.conf.dim_image_feat)
            assert(feat_arr_te.shape[1]==self.conf.dim_image_feat)
            logger.info('Arranged image features loaded')
        
        # Finding NN images
        if self.conf.cal_distance_all:
            dis_trte = distance.cdist(feat_arr_te, feat_arr_tr, self.conf.distance_metric)
            assert(dis_trte.dtype == np.float64)
            logger.info('Distance calculation finished!')
            self.NNimg_list = np.argsort(dis_trte, axis=1)[:, :self.conf.num_NNimg].tolist()
        else:
            NNimg_list = []
            for ind_t in xrange(feat_arr_te.shape[0]):
                feat_te_tmp = feat_arr_te[ind_t, :].reshape((1, self.conf.dim_image_feat) )
                dis_trte_tmp = distance.cdist(feat_te_tmp, feat_arr_tr, self.conf.distance_metric)
                NNimg_list.append(np.argsort(dis_trte_tmp, axis=1)[0, :self.conf.num_NNimg].tolist() )
                if (ind_t + 1) % self.conf.num_show_finished == 0:
                    logger.info('%d NN image finished' % (ind_t + 1) )
            self.NNimg_list = NNimg_list
            
        np.save(fname_NNimg_list, self.NNimg_list)
        logger.info('Find NN image finished')
    
    def consensus_rerank(self, method='cider', flag_eval=True):
        # Only support cider and bleu currently
        assert(method=='cider' or method=='bleu')
        assert(len(self.NNimg_list)==len(self.anno_list_hypo) )
        fol_cache = os.path.join(self.conf.fol_root_cache, self.conf.name_cache)
        if not os.path.exists(fol_cache):
            os.makedirs(fol_cache)
        
        if method=='cider':
            # prepare cider
            coco = COCO(self.conf.fname_eval_ref)
            cocoEvalCider = COCOEvalCapPairCider(coco)
            cocoEvalCider.setup()
            k = self.conf.k_cider
            m = self.conf.m_cider
            key_reranking = 'rerank_%s_k%d_m%d_cider' \
                % (self.conf.gen_method, k, m)
        else:
            k = self.conf.k_bleu
            m = self.conf.m_bleu
            key_reranking = 'rerank_%s_k%d_m%d_bleu' \
                % (self.conf.gen_method, k, m)
        
        # start reranking
        rerank_ind = {} # the ind is used to rerank the sentences that are in original order, for sGPN\dagger in grounding evaluation
        if self.anno_list_reranked == []:
            anno_list_reranked = self.anno_list_hypo
        else:
            anno_list_reranked = self.anno_list_reranked
            
        for (ind_te, anno) in enumerate(anno_list_reranked): # anno: a dict for an image, 10 sentences are in 'gen_beam_search_10'
            sentences_gen = anno[self.conf.gen_method]
            sentences_ret = []
            for ind_NN in xrange(k):
                ind_tr = self.NNimg_list[ind_te][ind_NN]
                sentences_ret += self.anno_list_ref[ind_tr]['sentences']
            sim = []
            for (ind_g, sen_gen) in enumerate(sentences_gen):
                b_s_arr = []
                for (ind_r, sen_ret) in enumerate(sentences_ret):
                    if method == 'cider':
                        b_s_arr.append(cocoEvalCider.calculate_cider_sentence( \
                            ' '.join(sen_gen), ' '.join(sen_ret) ) )
                    else:
                        b_s_arr.append(bs_util.calculate_bleu_sentence( \
                            sen_gen, sen_ret, self.conf.bleu_ngram, fpr=self.conf.fpr_bleu) )
                b_s_arr.sort(reverse=True)
                sim.append(sum(b_s_arr[:m]))
        
            # Sort the sentence according to sim
            arg_sim = np.argsort(-np.array(sim)).tolist()
            anno[key_reranking] = [sentences_gen[x] for x in arg_sim] # put the sentences in the order of ranking score, each sentence is decompose to a list of words
            rerank_ind[anno['id']] = arg_sim

            if (ind_te + 1) % self.conf.num_show_finished == 0:
                logger.info('%d image reranking finished' % (ind_te + 1) )

        #np.save(self.conf.fname_hypo_list.split("/")[-1].split(".")[0] + "_rerank_ind.npy", rerank_ind)
        self.anno_list_reranked = anno_list_reranked # list of dicts, each dict is an image
        fname_anno_list_reranked = os.path.join(fol_cache, 'anno_list_hypo_rerank_%s_%s.npy' \
            % (self.conf.name_feat, self.conf.distance_metric) )
        np.save(fname_anno_list_reranked, anno_list_reranked) # save the sentences in the order of reranked, in key of 'rerank_caption_k60_m125_cider'
        
        # Write the statistics to hard disk and evaluate the performance
        if flag_eval:
            fname_coco_json = os.path.join(fol_cache, 'coco_json_%s_%s_%s.json' \
                % (self.conf.name_feat, self.conf.distance_metric, method) )
            fout_eval = open(os.path.join(fol_cache, 'eval_stat_%s_%s_%s.txt' \
                % (self.conf.name_feat, self.conf.distance_metric, method) ), 'w')
            self._anno_genS2coco(anno_list_reranked, key_reranking, 0, fname_coco_json)  # 0: only save top-1 sentence after re-ranking
                
            coco = COCO(self.conf.fname_eval_ref)
            cocoRes = coco.loadRes(fname_coco_json)
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.params['image_id'] = cocoRes.getImgIds()
            cocoEval.evaluate()
                
            # print output evaluation scores
            for metric, score in cocoEval.eval.items():
                print '%s: %.3f'%(metric, score)
                print >>fout_eval, '%s: %.3f'%(metric, score)
            fout_eval.close()
            
            self.cocoEval = cocoEval
            
    def _anno_genS2coco(self, anno_list, key, ind_used, fname_COCO_json):
        anno_arr_coco = []
        for anno in anno_list:
            gen_sentence = anno[key][ind_used]  # 0: only save top-1 sentence after re-ranking
            anno_coco = dict()
            anno_coco['image_id'] = anno['id']
            anno_coco['caption'] = ' '.join(gen_sentence) + '.'
            anno_arr_coco.append(anno_coco)
        json.dump(anno_arr_coco, open(fname_COCO_json, 'w') )
