# -*- coding: utf-8 -*-
'''
Main script of the sentence level bleu scores

Authors: Junhua Mao <mjhustc@ucla.edu>
'''

import math
import numpy as np

#from nltk.compat import Counter
from collections import Counter
from nltk.util import ngrams

def calculate_bleu_sentence(target_sentence, ref_sentence, n, fpr=1.0):
    '''
    target_sentences is a list of words for the target sentence
    ref_sentences is a list of ref sentences
    n is the ngram we adopt
    '''
    
    fscores = []
    precisions = []
    recalls = []

    for i in range(n):
        if n > len(target_sentence) or n > len(ref_sentence):
            fscores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        else:
            counts = Counter(ngrams(target_sentence, i+1))
            ref_counts = Counter(ngrams(ref_sentence, i+1))
            max_counts = {}
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0), ref_counts[ngram])
            clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())
            target_num = sum(counts.values())
            ref_num = sum(ref_counts.values())
            acc_num = sum(clipped_counts.values())
            
            if acc_num == 0:
                fscores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                pre = float(acc_num) / float(target_num)
                rec = float(acc_num) / float(ref_num)
                f = ((fpr + 1) * pre * rec) / (rec + fpr * pre)
                fscores.append(f)
                precisions.append(pre)
                recalls.append(rec)

    B_s = []
        
    for i in range(n):
        weighted_s = math.fsum(fscores[:i+1]) / float(i+1)
        B_s.append(weighted_s)
    
    return B_s[n-1]
