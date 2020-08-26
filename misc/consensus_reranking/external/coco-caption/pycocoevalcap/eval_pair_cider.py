# JMao: Movidfined from eval.py from https://github.com/tylin/coco-caption
# Used to calcuate cider distance between two sentences

from tokenizer.ptbtokenizer import PTBTokenizer
from cider.cider_scorer_compute_sentence import CiderScorer

class COCOEvalCapPairCider:
    def __init__(self, coco):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.params = {'image_id': coco.getImgIds()}
        self.flag_setup = False
        self.cider_scorer = None

    def setup(self):
        imgIds = self.params['image_id']

        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.coco.imgToAnns[imgId]

        # Tokenize
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        
        # print len(gts.keys), len(res.keys)

        # Setup cider
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer()

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)
            
        cider_scorer.compute_doc_freq()
        assert(len(cider_scorer.ctest) >= max(cider_scorer.document_frequency.values()))
        self.flag_setup = True
        self.cider_scorer = cider_scorer
        
    def calculate_cider_sentence(self, sen, ref):
        assert(self.flag_setup)
        return self.cider_scorer.compute_cider_sen_pair(sen, ref)
