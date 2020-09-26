import numpy as np
import pickle

from ptbtokenizer import PTBTokenizer
from bleu import Bleu
import argparse

np.random.seed(2019)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='captions_60000.npy',
                help='the input file that contains the generated captions')
parser.add_argument('--evaluate_mB4', action='store_true',
                help='if true, evaluate mBLEU4 which takes much longer time than other metrics')
args = parser.parse_args()

input_file = args.input_file
evaluate_mB4 = args.evaluate_mB4
split = 'M-RNN'
metric = [1,2,3,4]
"""
metric 1: Distinct Caption
metric 2: Novel Caption
metric 3: 1-gram and 2-gram
metric 4: mBLEU-4
"""
tokenizer = PTBTokenizer()

def setImgToEvalImgs(scores, imgIds, method, imgToEval):
    for imgId, score in zip(imgIds, scores):
        if not imgId in imgToEval:
            imgToEval[imgId] = {}
            imgToEval[imgId]["image_id"] = imgId
        imgToEval[imgId][method] = score

def cal_avg_B4(custom_gts, custom_res):
    # input tested senetences, and (top_N - 1) corresponding 'gt' sentences
    # return the BLEU-4 score
    # calculate BLEU scores in tradictional way
    gts  = tokenizer.tokenize(custom_gts)
    res = tokenizer.tokenize(custom_res)
    print('setting up scorers...')
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
    imgToEval = {}
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        if type(method) == list:
            score, scores, subgraph_training_bleu = scorer.compute_score(gts, res)
            for sc, scs, m in zip(score, scores, method):
                setImgToEvalImgs(scs, list(gts.keys()), m, imgToEval)
                print("%s: %0.3f"%(m, sc))
    B_4s = [imgToEval[sen_id]['Bleu_4'] for sen_id in custom_gts.keys()]
    return B_4s

if __name__ == '__main__':
    if 4 in metric and evaluate_mB4:
        subgraph_sents = np.load(input_file, allow_pickle=True,encoding='latin1').tolist()
        top_N = [20, 100]  # randomly select 20 or 100 sentences and use best-5 after reranking for diversity score
        img_b4 = [[],[]]
        for img_i, item in enumerate(subgraph_sents):
            sub_num = len(item['caption'])
            for t_i, top_k in enumerate(top_N):
                rand_ind = np.random.choice(sub_num, min(top_k,sub_num), replace=False)
                selected_gpn = item['subgraph_score'][rand_ind]
                selected_sents_ind = rand_ind[np.argsort(selected_gpn)[::-1][:5]]  # get best-5 after re-ranking
                #selected_sents_ind = rand_ind[:5]  # random 5 sentences
                selected_sents_list = [item['caption'][ind] for ind in selected_sents_ind]
                this_img_b4 = []
                for sen_i, sen in enumerate(selected_sents_list):
                    custom_res = {}
                    custom_gts = {}
                    this_id = str(item['image_id']) + '_' + str(sen_i)
                    custom_res[this_id] = [{'caption': sen}]  # tested sentence
                    custom_gts[this_id] = [] # make up 'gt' sentences of tested sentence
                    for gt_j, gt_sen in enumerate(selected_sents_list): 
                        if gt_j != sen_i: # excluding itself
                            custom_gts[this_id].append({'caption': gt_sen})
                    this_img_b4.extend(cal_avg_B4(custom_gts, custom_res))
                img_b4[t_i].append(np.mean(np.array(this_img_b4)))

        print('\nm-BLEU-4 for best-5 out of random 20 sentences: {}'.format(np.mean(np.array(img_b4[0]))))
        print('m-BLEU-4 for best-5 out of random 100 sentences: {}'.format(np.mean(np.array(img_b4[1]))))

    if 3 in metric:
        subgraph_sents = np.load(input_file, allow_pickle=True,encoding='latin1').tolist()
        top_N = [20, 100]  # randomly select 20 or 100 sentences and use best-5 after reranking for diversity score
        n_gram = np.zeros((len(top_N), 2, len(subgraph_sents))) # 1-gram and 2-gram

        for img_i, item in enumerate(subgraph_sents):
            sub_num = len(item['caption'])
            for t_i, top_k in enumerate(top_N):
                rand_ind = np.random.choice(sub_num, min(top_k,sub_num), replace=False)
                selected_gpn = item['subgraph_score'][rand_ind]
                selected_sents_ind = rand_ind[np.argsort(selected_gpn)[::-1][:5]]  # get best-5 after re-ranking
                #selected_sents_ind = rand_ind[:5]  # random 5 sentences
                selected_sents_list = [item['caption'][ind] for ind in selected_sents_ind]
                split_sents_list = [sent.split(' ') for sent in selected_sents_list]
                words_list = [wd for l in split_sents_list for wd in l] # 1-gram
                total_wd = len(words_list)
                # count distinct grams per caption
                gram_per_cap = [[],[]]; gram_per_cap[0] = words_list
                for sent in split_sents_list:
                    # get 2-grams per sentence
                    for w_ind in range(len(sent)):
                        if w_ind != len(sent) - 1:  
                            gram_per_cap[1].append((sent[w_ind], sent[w_ind+1])) # 2-gram
                n_gram[t_i, 0, img_i] = len(set(gram_per_cap[0])) / float(total_wd)
                n_gram[t_i, 1, img_i] = len(set(gram_per_cap[1])) / float(total_wd)
        
        print('\n1-gram diversity for best-5 out of random 20 sentences: {}'.format(np.mean(n_gram[0,0])))
        print('2-gram diversity for best-5 out of random 20 sentences: {}'.format(np.mean(n_gram[0,1])))
        print('1-gram diversity for best-5 out of random 100 sentences: {}'.format(np.mean(n_gram[1,0])))
        print('2-gram diversity for best-5 out of random 100 sentences: {}'.format(np.mean(n_gram[1,1])))

    if 2 in metric:
        if split == 'karpathy':
            coco_names = "karpathy_all_images.txt"
            with open(coco_names,'r') as f:
                coco_img_ids = []
                for item in f:
                    coco_img_ids.append(item.split(' ')[1].strip())
            coco_img_ids = coco_img_ids[10000:]
        elif split == 'M-RNN':
            MRNN_split_dict = np.load('../../data/MRNN_split_dict.npy', allow_pickle=True,encoding='latin1').tolist()
            coco_img_ids = [str(k) for k,v in MRNN_split_dict.items() if v == 'train']

        all_cap_dict = pickle.load(open('all_caption_dict.pkl','rb'))
        train_sents = []
        for img_id in coco_img_ids:
            train_sents.extend([sen.lower().replace('.','') for sen in all_cap_dict[img_id]])
        train_sents = set(train_sents)

        subgraph_sents = np.load(input_file, allow_pickle=True,encoding='latin1').tolist()
        top_N = [20, 100]  # randomly select 20 or 100 sentences and use best-5 after reranking for diversity score
        novel_cnt = [0, 0]
        for img_i, item in enumerate(subgraph_sents):
            sub_num = len(item['caption'])
            for t_i, top_k in enumerate(top_N):
                rand_ind = np.random.choice(sub_num, min(top_k,sub_num), replace=False)
                selected_gpn = item['subgraph_score'][rand_ind]
                selected_sents_ind = rand_ind[np.argsort(selected_gpn)[::-1][:5]]  # get best-5 after re-ranking
                #selected_sents_ind = rand_ind[:5]  # random 5 sentences
                this_sents_list = [item['caption'][ind] for ind in selected_sents_ind]
                n_sent = [1 for sen_i in this_sents_list if sen_i not in train_sents]
                novel_cnt[t_i] += len(n_sent)
        print('\nNovel Caption for best-5 out of random 20 sentences: {}'.format(novel_cnt[0]))
        print('Novel Caption count for best-5 out of random 100 sentences: {}'.format(novel_cnt[1]))

    if 1 in metric:
        subgraph_sents = np.load(input_file, allow_pickle=True,encoding='latin1').tolist()
        top_N = [20, 100] # randomly select 20 or 100 sentences and use best-5 after reranking for diversity score
        uniqueness = np.zeros((len(top_N), len(subgraph_sents)))

        for img_i, item in enumerate(subgraph_sents):
            sub_num = len(item['caption'])
            for t_i, top_k in enumerate(top_N):
                rand_ind = np.random.choice(sub_num, min(top_k,sub_num), replace=False)
                this_sents_list = [item['caption'][ind] for ind in rand_ind]
                this_sents_set = set(this_sents_list)
                uniqueness[t_i, img_i] = len(this_sents_set) / float(len(this_sents_list))

        print('\nDistinct Caption of random-20 sentences: {}'.format(np.mean(uniqueness[0])))
        print('Distinct Caption of random-100 sentences: {}'.format(np.mean(uniqueness[1])))

