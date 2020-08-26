'''
Setting up folder and file dirs, and method name

Authors: Junhua Mao <mjhustc@ucla.edu>
'''
import os

# resnet-101
class Conf_Cr:
    def __init__(self, input_args):
        # Settings for image features
        self.fol_image_feat = './image_features_mRNN'
        self.name_feat = '101'
        self.dim_image_feat = 2048
        if input_args.dataset == 'coco': 
            feat_name = 'res_feat_%s_dct_mscoco_2014.npy' % '101'
        elif input_args.dataset == 'flickr30k':
            feat_name = 'res_feat_%s_dct_flickr30k.npy' % '101'
        self.fname_dct_feat = os.path.join(self.fol_image_feat, feat_name)

        # Settings for annotation list file
        self.fol_anno_list = './mscoco_anno_files'
        if input_args.dataset == 'coco':
            if input_args.split == 'karpathy': 
                anno_name = 'karpathy_train_val_anno_list.npy'
            elif input_args.split == 'MRNN':
                anno_name = 'anno_list_mscoco_trainModelVal_m_RNN.npy'
        elif input_args.dataset == 'flickr30k':
                anno_name = 'flickr30k_karpathy_train_val_anno_list.npy'
        self.fname_anno_list_ref = os.path.join(self.fol_anno_list, anno_name) 
        
        # Settings for hypotheses annotation list file
        self.fol_hypo_list = './hypotheses_mRNN'
        #self.name_tar_data = 'crVal'
        self.fname_hypo_list = os.path.join(self.fol_hypo_list, input_args.input_file)
        self.gen_method = 'caption'

        # self.fname_hypo_list = os.path.join(self.fol_hypo_list, 'test_20_genS.npy') # DEBUG
        
        # Settings for hyperparamters (validated on crVal)
        # See more details of them in the paper
        self.distance_metric = 'euclidean'
        self.k_cider = 60
        self.m_cider = 125
        self.k_bleu = 60
        self.m_bleu = 175
        self.fpr_bleu = 1.0
        self.bleu_ngram = 4
        
        # Memory settings
        self.num_NNimg = 1000 # Only keep neatest num_NNimg images
        self.del_useless = True # Delete useless memory
        self.cal_distance_all = False # Setting this to True will accelerate the process
                                      # But requires much more memory
        
        # Cache settings
        self.name_cache = 'res101'
        self.fol_root_cache = input_args.output_folder
        print("\nInput file is {}\n".format(self.fname_hypo_list))
        print("\nOutput results to folder {}\n".format(self.fol_root_cache))

        # Debug settings
        self.num_show_finished = 200
        
        # Evaluation settings
        self.fol_coco_eval = './external/coco-caption'
        if 'flickr30k' in self.fname_anno_list_ref:
            self.fname_eval_ref = os.path.join('../../coco-caption', \
                'annotations', 'caption_flickr30k.json')
        else:
            self.fname_eval_ref = os.path.join('../../coco-caption', \
                'annotations', 'captions_val2014.json')