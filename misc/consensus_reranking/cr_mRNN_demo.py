import numpy as np
import json
import argparse

np.random.seed(2019)
sentence_path = 'hypotheses_mRNN'
gt_sentence_path = 'mscoco_anno_files'
img_feat_path = 'image_features_mRNN'

if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=4,
                    help='top k captions selected by sGPN scores are used for consensus reranking')
    parser.add_argument('--only_consensus', action='store_true',
                    help='if only consensus, apply consensus on random-k captions')
    parser.add_argument('--rand_k', type=int, default=20,
                    help='randomly select k captions from all generated captions for consensus reranking, \
                          20 or 100')
    parser.add_argument('--input_file', type=str, default='captions_60000.npy',
                    help='the caption file contains the generated captions')
    parser.add_argument('--dataset', type=str, default='coco',
                    help='the dataset to be evaluated on, coco or flickr30k')
    parser.add_argument('--split', type=str, default='MRNN',
                    help='the dataset to be evaluated on, MRNN or karpathy split for coco, \
                          only karpathy split on flickr30k')
    parser.add_argument('--converted_file', type=str, default='splitted_captions_60000_top_4.npy',
                    help='converted file name, the file contains the splitted tokens of captions')
    parser.add_argument('--output_folder', type=str, default='output',
                    help='the folder that contains all output files')
    args = parser.parse_args()

    # put the predicted sentences by subgraphs into the format of M-RNN repo 
    caption_file = args.input_file
    select_top = not args.only_consensus
    if select_top:
        top_k = args.top_k
        converted_file = 'splitted_{}_top_{}.npy'.format(caption_file.split('.')[0],top_k)
    else:
        rand_k = args.rand_k
        converted_file = 'splitted_{}_random_{}.npy'.format(caption_file.split('.')[0],rand_k)

    subgraph_sents = np.load(sentence_path + '/' + caption_file,allow_pickle=True,encoding='latin1').tolist()
    split_sents = []
    for item in subgraph_sents:
        this_dict = {}
        this_dict['id'] = item['image_id']
        this_dict['caption'] = []
        if select_top:  # select top sGPN score sentence
            this_k = min(top_k, len(item['caption']))
            for sen_i in range(this_k): 
                this_dict['caption'].append(item['caption'][sen_i].split(' '))
        else:  # randomly select sentences/subgraphs
            this_k = min(rand_k, len(item['caption']))
            rand_ind = np.random.choice(len(item['caption']), this_k, replace=False)
            for sen_i in rand_ind: 
                this_dict['caption'].append(item['caption'][sen_i].split(' '))            
        split_sents.append(this_dict)
    np.save(sentence_path + '/' + converted_file, split_sents)
    args.converted_file = converted_file
    args.output_folder = 'output_' + caption_file.split('.')[0]

    # apply consensus reranking
    from concensus_reranking_utils.consensus_reranking import Consensus_Reranking
    import time
    start = time.time()

    # Create an CR class
    cr_mRNN = Consensus_Reranking(args)
    # Load anno file list
    cr_mRNN.load_anno_ref_hypo()
    # Find NN image
    cr_mRNN.find_NNimg()
    # Consensus Rerank
    #cr_mRNN.consensus_rerank(method='bleu')
    cr_mRNN.consensus_rerank(method='cider')

    print("\n\nTotally used time: {}".format(time.time()-start))
