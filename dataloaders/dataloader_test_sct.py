from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data
import json
import h5py
import os
import numpy as np
import random
import copy

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    The loading method depend on extention.
    """
    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x,encoding='latin1')
        else:
            if "sg" or "graph" in db_path:
                self.loader = lambda x: np.load(x,allow_pickle=True,encoding='latin1')['feat'].tolist()  # SG output, should be a dict
            else:
                self.loader = lambda x: np.load(x,encoding='latin1')['feat']
        self.db_type = 'dir'
    
    def get(self, key):
        f_input = os.path.join(self.db_path, key + self.ext)

        # load image
        feat = self.loader(f_input)

        return feat

class pklLoader:
    def __init__(self, db_path):
        self.db_path = db_path
        self.loader = lambda x: np.load(x,encoding='latin1')['feat']

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        if 'flickr' in opt.input_label_h5:
            dataset_name = 'flickr30k'
        #elif 'coco' in opt.input_label_h5:
            #dataset_name = 'COCO'

        mask_version = '1000'
        self.use_greedy_subg = opt.use_greedy_subg
        self.use_gt_subg = opt.use_gt_subg
        self.trip_loader = HybridLoader("data/{}_sg_output_64".format(dataset_name), ".npz") 
        self.subgraph_mask = HybridLoader("data/{}_gt_graph_mask".format(dataset_name), ".npz")
        self.sct_dict = np.load('data/sct_dict_test_grouped_gt_box.npy',allow_pickle=True,encoding='latin1').tolist()
        self.img_wh = np.load('data/flickr30k_img_wh.npy',allow_pickle=True,encoding='latin1').tolist()
        
        # load in the sequence data
        #self.label = self.h5_label_file['labels'][:]
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train': 
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test': 
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

        self.obj_num = opt.obj_num 
        self.rel_num = opt.rel_num 
        self.half_mini_batch = None 

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # let's first use #ncap captions
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            seq[:ncap,:] = self.h5_label_file['labels'][ix1: ix1 + ncap, :self.seq_length]
            # then pad the others by subsampling (with replacement)
            for q in range(seq_per_img)[ncap:]:
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = ix1 #random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] 
        att_batch = [] 
        label_batch = [] 
        trip_pred_batch = [] 
        obj_dist_batch = []
        obj_box_batch = []
        rel_ind_batch = []
        pred_fmap_batch = []
        pred_dist_batch = []

        wrapped = False
        data = {}
        
        tmp_list = self._prefetch_process[split].get()  # call one time to get a whole batch instead of fetching one by one instance
        wrapped = tmp_list[-1]
        tmp_list = tmp_list[:-1]

        # merge features
        data['trip_pred'] = None 
        data['pred_fmap'] = None 

        # torch tensors in pin_memory, fetch from Blobfetcher.get()
        tmp_fc, tmp_att, tmp_object_dist, tmp_rel_ind, tmp_pred_dist, tmp_label, tmp_masks, tmp_ix, \
        tmp_gpn_obj_ind, tmp_gpn_pred_ind, tmp_gpn_nrel_ind, tmp_att_mask, tmp_gpn_pool_mtx, tmp_this_mini_batch = tmp_list
        data['fc_feats'] = tmp_fc.view(-1, 2048)
        data['att_feats'] = tmp_att.view(-1, self.obj_num, 2048)
        data['obj_dist'] = tmp_object_dist.view(-1, self.obj_num, 1599)
        data['rel_ind']= tmp_rel_ind.view(-1, self.rel_num, 2)
        data['pred_dist'] = tmp_pred_dist.view(-1, self.rel_num, 21)
        data['labels'] = tmp_label.view(-1, self.seq_length + 2)
        data['masks'] = tmp_masks.view(-1, self.seq_length + 2)
        data['att_masks'] = tmp_att_mask.view(-1,2,tmp_this_mini_batch, self.obj_num) #None # mask: the valid region numbers in some images are less than max_att_len in current batch
        data['gpn_obj_ind'] = tmp_gpn_obj_ind.view(-1,2,tmp_this_mini_batch,self.obj_num)
        data['gpn_pred_ind'] = tmp_gpn_pred_ind.view(-1,2,tmp_this_mini_batch,self.rel_num)
        data['gpn_nrel_ind'] = tmp_gpn_nrel_ind.view(-1,2,tmp_this_mini_batch,self.rel_num,2)
        data['gpn_pool_mtx'] = tmp_gpn_pool_mtx.view(-1,2,tmp_this_mini_batch,self.obj_num,self.obj_num)
        data['obj_box'] = None

        # batch data not in pin_memory, which stays as list
        data['gts'] = [] # all ground truth captions of each images
        data['infos'] = []
        for ix in tmp_ix.view(-1).numpy():
            data['gts'].append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            data['infos'].append({'ix':ix, 'id':self.info['images'][ix]['id'], 'file_path':self.info['images'][ix]['file_path']})

        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}

        return data

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
     
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
     
        # return the intersection over union value
        return iou

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index
        this_ix = np.array([index])
        img_id = self.info['images'][ix]['id']
        ############### load full SG with dummy node and dummpy predicate/rel_ind  ###############
        # object related data 
        sg_output = self.trip_loader.get(str(img_id))
        object_fmap = sg_output['object_fmap'][:self.obj_num,:]
        object_dist = sg_output['object_dist'][:self.obj_num,:]

        pad_object_fmap = np.full((1, self.obj_num, object_fmap.shape[1]), 0).astype('float32') # pad with the dummy/empty node
        pad_object_dist = np.concatenate((np.ones((1, self.obj_num, 1)), np.zeros((1, self.obj_num, object_dist.shape[1]-1))), axis=2).astype('float32')
        fc_feat = np.full((1,object_fmap.shape[1]), 0).astype('float32')

        pad_object_fmap[0,:(self.obj_num - 1),:] = object_fmap
        pad_object_dist[0,:(self.obj_num - 1),:] = object_dist

        # 3. predicate related data
        pred_dist = sg_output['pred_dist']
        rel_ind = sg_output['rel_ind']
        pad_rel_ind = np.full((1, self.rel_num, rel_ind.shape[1]), self.obj_num-1) # pad the rel_ind with the dummy/empty node index
        pad_pred_dist = np.concatenate((np.ones((1, self.rel_num, 1)), np.zeros((1, self.rel_num, pred_dist.shape[1]-1))), axis=2).astype('float32')

        this_len = min(rel_ind.shape[0], self.rel_num - 1)
        pad_pred_dist[0, :this_len,:] = pred_dist[:this_len]
        pad_rel_ind[0, :this_len,:] = rel_ind[:this_len]
        ############### load full SG with dummy node and dummpy predicate/rel_ind  ###############

        ############### flickr30k show control and tell  ###############
        w, h = self.img_wh[img_id]
        object_box = sg_output['boxes'][:self.obj_num,:]  # bbox from SG detector
        sg_boxes = object_box * max(w, h) / 592  # scale to image size
        rel_ind = sg_output['rel_ind'].astype(int)
        object_cls = np.argmax(object_dist, axis=1)
        
        # find the matched detection boxes
        region_sets = self.sct_dict[str(img_id)].numpy()
        all_match_ind = []
        for i in range(region_sets.shape[0]):  # per sentence
            valid_num = region_sets[i][:,4].nonzero()[0].shape[0]
            this_set = region_sets[i][:valid_num,:4]
            this_set_matched = []
            for j in range(this_set.shape[0]):  # per SCT grounding box
                this_region = this_set[j]
                max_iou = 0
                matched_ind = -1
                for k in range(sg_boxes.shape[0]):  # per detection box
                    this_iou = self.bb_intersection_over_union(this_region, sg_boxes[k])
                    if this_iou > max_iou:
                        matched_ind = k
                        max_iou = this_iou
                if matched_ind != -1:
                    this_set_matched.append([matched_ind,max_iou])

            # filter the low iou match
            filtered_match_ind = [m_i[0] for m_i in this_set_matched if m_i[1] >= 0.5]
            if len(filtered_match_ind) == 0: # if no match higher than IoU_thres, then only keep the best matched box
                adaptive_iou = np.max([m_i[1] for m_i in this_set_matched])
                if adaptive_iou == 0: # no overlapping at all, then keep all SG nodes
                    filtered_match_ind = np.arange(sg_boxes.shape[0])
                else:
                    filtered_match_ind = [m_i[0] for m_i in this_set_matched if m_i[1] >= adaptive_iou]

            all_match_ind.append(filtered_match_ind)
        
        # extract / select sub-graphs according to the matched nodes (boxes)
        subg_num = len(all_match_ind)
        this_mini_batch = subg_num

        gpn_obj_ind = np.full((self.seq_per_img,2,this_mini_batch,self.obj_num),self.obj_num-1)
        gpn_att_mask = np.full((self.seq_per_img,2,this_mini_batch,self.obj_num), 0).astype('float32')
        gpn_pred_ind = np.full((self.seq_per_img,2,this_mini_batch,self.rel_num),self.rel_num-1)
        gpn_nrel_ind = np.full((self.seq_per_img,2,this_mini_batch,self.rel_num,2), self.obj_num-1)
        gpn_pool_mtx = np.zeros((self.seq_per_img,2,this_mini_batch,self.obj_num,self.obj_num)).astype('float32')
        
        if self.use_gt_subg:
            subgraph_dict = self.subgraph_mask.get(str(img_id))
            mask_info = subgraph_dict['subgraph_mask_list']
            gt_seed_node_list = [mask[4] for mask in mask_info[:5]]  # pre-computed gt seed nodes
            node_iou_mtx = subgraph_dict['node_iou_mtx'][:,5:]
        
        for i in range(subg_num):
            if self.use_greedy_subg: # greedily extract the sub-graphs using the nodes matched by grounding boxes
                seed_node = all_match_ind[i]
                keep_obj_ind = np.array(seed_node)
                keep_obj_mask = np.zeros(object_cls.shape[0]).astype(int)
                keep_obj_mask[keep_obj_ind] = 1

                # keep the nodes that have same class with seed nodes
                kept_cls = np.unique(object_cls[keep_obj_mask == 1])
                for k in range(keep_obj_mask.shape[0]):
                    if object_cls[k] in kept_cls:
                        keep_obj_mask[k] = 1
                keep_obj_ind = keep_obj_mask.nonzero()[0]

                # the predicates btw matched nodes and their neighbors
                keep_rel_mask = np.zeros(rel_ind.shape[0]).astype(int)
                keep_rel_byte = np.zeros(rel_ind.shape[0]) == 1
                # include neighbor nodes in sub-graphs
                for k in range(keep_obj_ind.shape[0]):
                    keep_rel_byte = keep_rel_byte | (rel_ind[:,0] == keep_obj_ind[k]) | (rel_ind[:,1] == keep_obj_ind[k])
                keep_rel_mask[keep_rel_byte] = 1 

                # update kept nodes
                keep_obj_mask[np.unique(rel_ind[keep_rel_mask == 1])] = 1
                keep_obj_ind = keep_obj_mask.nonzero()[0]    

                # new_rel_ind
                new_rel_ind = copy.deepcopy(rel_ind)
                for k in range(keep_obj_ind.shape[0]):
                    new_rel_ind[np.where(rel_ind == keep_obj_ind[k])] = k
                new_rel_ind = new_rel_ind[keep_rel_mask == 1] 
                
                # make up the input based on the subgraph
                tmp = (keep_obj_mask==1).nonzero()[0]
                gpn_obj_ind[:,:,i,:tmp.shape[0]] = tmp
                gpn_att_mask[:,:,i,:tmp.shape[0]] = 1
                gpn_pool_mtx[:,:,i,np.arange(tmp.shape[0]),np.arange(tmp.shape[0])] = 1

                tmp = (keep_rel_mask==1).nonzero()[0]
                gpn_pred_ind[:,:,i,:tmp.shape[0]] = tmp

                tmp = new_rel_ind
                gpn_nrel_ind[:,:,i,:tmp.shape[0]] = tmp
            elif self.use_gt_subg: # search the index of gt sub-graphs since the order is shuffled by grouped gt captions in SCT
                seed_node = np.array(all_match_ind[i])
                matched_sent_ind = None
                # 1. find the sentence index by comparing the matched nodes with gt subgraphs
                for sn_i,pre_com_sn in enumerate(gt_seed_node_list): 
                    uq_pre_com_sn = np.unique(pre_com_sn)  # seed node from gt sub-graphs in conventional captioning order 
                    uq_sn = np.unique(seed_node) # seed node from grouped sentences in SCT
                    if uq_sn.shape[0] == uq_pre_com_sn.shape[0]:
                        if len((uq_pre_com_sn - uq_sn).nonzero()[0]) == 0: # matched if exactly the same because of same box matching processing
                            matched_sent_ind = sn_i
                            break 
                # 2. select corresponding gt sub-graph             
                subg_ind = matched_sent_ind

                # 3. make up the input based on the gt subgraph
                tmp = mask_info[subg_ind][1].nonzero()[0]
                gpn_obj_ind[:,:,i,:tmp.shape[0]] = tmp
                gpn_att_mask[:,:,i,:tmp.shape[0]] = 1
                gpn_pool_mtx[:,:,i,np.arange(tmp.shape[0]),np.arange(tmp.shape[0])] = 1

                tmp = mask_info[subg_ind][2].nonzero()[0]
                gpn_pred_ind[:,:,i,:tmp.shape[0]] = tmp

                tmp = mask_info[subg_ind][3]
                gpn_nrel_ind[:,:,i,:tmp.shape[0]] = tmp                           
            else:
                print("Not implemented!")           
        ############### flickr30k show control and tell  ###############

        label = np.zeros([self.seq_per_img, self.seq_length + 2], dtype = 'int64')
        label[:, 1 : self.seq_length + 1] = self.get_captions(ix, self.seq_per_img)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label)))
        mask_batch = np.zeros([label.shape[0], self.seq_length + 2], dtype = 'float32')
        for idx, row in enumerate(mask_batch):
            row[:nonzeros[idx]] = 1  # keep the 'start' + sentence + 'end', and mask out the rest

        # return as tmp in BlobFetcher.get()
        return fc_feat, pad_object_fmap, pad_object_dist, pad_rel_ind, pad_pred_dist, label, mask_batch, this_ix, \
        gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_att_mask, gpn_pool_mtx, np.array([this_mini_batch])

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.batch_size = dataloader.batch_size

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=self.batch_size,  # should same as the number in ri_next = ri + self.batch_size
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=self.dataloader.opt.num_workers))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False
        last_batch = False

        if self.split == 'train' or self.split == 'val':  # batch size >= 1, drop last batch
            ri = self.dataloader.iterators[self.split]  # count of images
            ix = self.dataloader.split_ix[self.split][ri]  # the index for train/val/test in the image list
            
            ri_next = ri + self.batch_size # should same as the number in "batch_size=self.batch_size,"
            if ri_next >= max_index:
                ri_next = 0
                if self.if_shuffle:
                    random.shuffle(self.dataloader.split_ix[self.split])
                wrapped = True
            
            self.dataloader.iterators[self.split] = ri_next  # shadow #data loaded by the dataloader 
            
            if wrapped is False and ri_next + self.batch_size >= max_index: # the next wrapped will be True, then current batch becomes last batch to be used
                last_batch = True
        elif self.split == 'test':  # batch size = 1, include all data
            ri = self.dataloader.iterators[self.split]  # count of images
            
            ri_next = ri + self.batch_size # should same as the number in "batch_size=self.batch_size,"
            if ri_next > max_index:
                ri_next = 0
                if self.if_shuffle:
                    random.shuffle(self.dataloader.split_ix[self.split])
                wrapped = True
            
            self.dataloader.iterators[self.split] = ri_next  # shadow #data loaded by the dataloader 
            
            if wrapped is False and ri_next + self.batch_size > max_index: # the next wrapped will be True, then current batch becomes last batch to be used
                last_batch = True
        else:
            assert False, "\nMode isn't correct! \n"

        return ri_next, wrapped, last_batch #ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()
        
        ix, wrapped, last_batch = self._get_next_minibatch_inds()
        
        if wrapped:  # drop the final incomplete batch
            self.reset()  # self.dataloader.iterators[self.split] has been reset to 0 before call self.reset(); enter the new epoch
            ix, wrapped, last_batch = self._get_next_minibatch_inds()  # shadow #data loaded by the dataloader 
            tmp = self.split_loader.next()
        else:
            tmp = self.split_loader.next()  # shadow #data loaded by the dataloader

        #assert tmp[-1][2] == ix, "ix not equal"
        # return to self._prefetch_process[split].get() in Dataloader.get_batch()

        if last_batch:  # last batch
            wrapped = True

        return tmp + [wrapped]