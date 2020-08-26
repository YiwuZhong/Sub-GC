#Annotation list files with hypotheses generated m-RNN

This folder contains files of the annotations with generated hypotheses by the mRNN model.

The data format is the same as the annotation files (see readme file at '../mscoco_anno_files/Readme.md').
We add a key of the annotation dictionary (e.g. 'gen_beam_search_10') of every image, whose value is the hypotheses for that image.

Currently we provide hypothese on our crVal set on MS COCO (i.e. 'anno_list_mscoco_crVal_m_RNN_beamsearch10.npy').
To conform to the rules of MS COCO evaluation server, we do not release the hypotheses on test2014 set.
But we release the images features and 1000 nearest neighbors on the test set.
You can use consensus reranking for your own method.
