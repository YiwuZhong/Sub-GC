#Annotation list files used in m-RNN

This folder contains annotation data of MS COCO dataset arranged and used in the [m-RNN project](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html).
The copyright of the annotations belongs to [MS COCO dataset](http://mscoco.org).

###Data format

Each file in this folder contains a list of python dictionary for a image in the MS COCO dataset.
The dictionary generally contains the following keys:

1. 'file_name': name of the image (orignial ms coco image name)
2. 'file_path': folder contains the image w.r.t. the ms coco image root
3. 'sentences': a list, each element of the list is a sentence which is conposed of a list of tokenized words (The annotation dictionaries in `ms_coco_anno_list_test2014.npy` does not have this key)
4. 'id': orignial coco image id
5. 'height': height of the image
6. 'width': width of the image
7. 'url': original url of the image
8. 'date_captured'
9. 'license'

You can use python command, such as 'numpy.load('FileName').tolist()' to load this file.

###File statistics

There are five python dictionary list files in the folder:

1. 'anno_list_mscoco_train_m_RNN.npy': List of 118,286 images used for training the m-RNN model.
2. 'anno_list_mscoco_modelVal_m_RNN.npy': List of 4,000 images used for validation the m-RNN model.
3. 'anno_list_mscoco_trainModelVal_m_RNN.npy': Combined list of (1) and (2) with 122,286 images. The consensus reranking will find the nearest neighbours from this set of images.
4. 'anno_list_mscoco_crVal_m_RNN.npy': List of 1,000 images used for tuning parameters for consensus reranking.
5. 'anno_list_mscoco_test2014.npy': List of 40,775 images in the MS COCO official testing set.

The image with id '167126' in the MS COCO official training set is partially broken so that we exclude it from our training set.
