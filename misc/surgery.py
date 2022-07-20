# create predictions from the other stuff
"""
Go from proposals + scores to relationships.
pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression
in all cases we'll return:
boxes, objs, rels, pred_scores
"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time
import ipdb

def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores, object_names, predicate_names, filename, obj_dists, obj_fmap, pred_fmap, generate_output, gt_classes, gt_rels):
    """
    Filters detections....
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores
    """
    if generate_output:
        if boxes.dim() != 2:
            raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

        num_box = boxes.size(0)
        assert obj_scores.size(0) == num_box

        assert obj_classes.size() == obj_scores.size()
        num_rel = rel_inds.size(0)
        assert rel_inds.size(1) == 2
        assert pred_scores.size(0) == num_rel
        num = 64  # maximum triplets in scene graph
        keep_all = False
        
        if keep_all:
            # save rel_ind (index for subject and object)
            output = {}
            output['rel_ind'] = rel_inds.cpu().numpy().astype('f')
            #output['pred_fmap'] = pred_fmap_sorted.astype('f')
            output['pred_dist'] = pred_scores.data.cpu().numpy().astype('f')
            output['object_fmap'] = obj_fmap.data.cpu().numpy().astype('f') # use bup repo feats directly
            output['object_dist'] = obj_dists.data.cpu().numpy().astype('f')
            output['boxes'] = boxes.data.cpu().numpy().astype('f')
            #with open("./sg_output_64/"+filename+".pkl","wb") as f:
                #pickle.dump(output, f, protocol=2)
            np.savez_compressed("./COCO_sg_output_all/"+filename,feat=output)            
        else:
            threshold = 0.75 # 0.5
            non_related = pred_scores.data[:,0]
            mask = torch.arange(pred_scores.size(0)).type_as(pred_scores.data).long()[non_related < threshold]
            if mask.dim() != 0:
                pred_scores = pred_scores[mask] 
                rel_inds = rel_inds[mask]
                #pred_fmap = pred_fmap[mask]
            else: # all predicate are weak where non-class scores are all higher than threshold
                num = 2 # make it at least a graph and has different sub-graph
            
            # after cleaning
            obj_scores0 = obj_scores.data[rel_inds[:,0]]
            obj_scores1 = obj_scores.data[rel_inds[:,1]]
            pred_scores_max, _ = pred_scores.data[:,1:].max(1)
            rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
            _, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

            # b. select first #num triplets
            rel_scores_idx = rel_scores_idx[:num]

            # after sorting
            rels = rel_inds[rel_scores_idx]
            pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()

            pred_scores_max_sorted, pred_classes_sorted = pred_scores[rel_scores_idx].data[:,1:].max(1)
            pred_scores_max_sorted = pred_scores_max_sorted.cpu().numpy()
            pred_classes_sorted = pred_classes_sorted + 1

            # save rel_ind (index for subject and object)
            output = {}
            output['rel_ind'] = rels.cpu().numpy().astype('f')
            #output['pred_fmap'] = pred_fmap_sorted.astype('f')
            output['pred_dist'] = pred_scores_sorted.astype('f')
            output['object_fmap'] = obj_fmap.data.cpu().numpy().astype('f') # use bup repo feats directly
            output['object_dist'] = obj_dists.data.cpu().numpy().astype('f')
            output['boxes'] = boxes.data.cpu().numpy().astype('f')
            #with open("./sg_output_64/"+filename+".pkl","wb") as f:
                #pickle.dump(output, f, protocol=2)
            np.savez_compressed("./COCO_sg_output_64_thres_075/"+filename,feat=output)
        
    else:
        if boxes.dim() != 2:
            raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

        num_box = boxes.size(0)
        assert obj_scores.size(0) == num_box

        assert obj_classes.size() == obj_scores.size()
        num_rel = rel_inds.size(0)
        assert rel_inds.size(1) == 2
        assert pred_scores.size(0) == num_rel

        obj_scores0 = obj_scores.data[rel_inds[:,0]]
        obj_scores1 = obj_scores.data[rel_inds[:,1]]

        pred_scores_max, pred_classes_argmax = pred_scores.data[:,1:].max(1)
        pred_classes_argmax = pred_classes_argmax + 1

        rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
        rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

        rels = rel_inds[rel_scores_idx].cpu().numpy()
        pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
        obj_scores_np = obj_scores.data.cpu().numpy()
        objs_np = obj_classes.data.cpu().numpy()
        boxes_out = boxes.data.cpu().numpy()

        return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted

