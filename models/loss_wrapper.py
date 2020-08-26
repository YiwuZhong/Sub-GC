import torch
import misc.utils as utils

"""
Loss wrapper to obtain loss for generated sentences
"""
class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = utils.LanguageModelCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices, trip_pred, \
                obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx):
        out = {}
        # cross entropy loss training
        lang_output, gpn_loss, subgraph_score = self.model(fc_feats, att_feats, labels, att_masks, trip_pred,\
            obj_dist, obj_box, rel_ind, pred_fmap, pred_dist,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx)
        if lang_output is not None:
            lang_loss = self.crit(lang_output, labels[:,1:], masks[:,1:])
        else:
            lang_loss = None

        out['gpn_loss'] = gpn_loss if gpn_loss is not None else None
        out['lang_loss'] = lang_loss
        return out
