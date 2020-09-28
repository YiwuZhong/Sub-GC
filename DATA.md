# Data Downloading

First download all the data from [Google Drive](https://drive.google.com/drive/folders/1mCx8R8d36ZpUSoVZKExs0FDA_IXiAiZA?usp=sharing). After that, extract the files from the compressed `*.zip` files and merge folders of `*_part1/`, `*_part2/` and `*_part3/` into a new folder `COCO_sg_output_64/`. The folder structure should be as follows:
```
├── data
│   ├── COCO_sg_output_64
│   │   ├── 9.npz
│   │   ├── ...
│   │   └── 581929.npz
│   ├── COCO_graph_mask_1000_rm_duplicate
│   │   ├── 9.npz
│   │   ├── ...
│   │   └── 581929.npz
│   ├── flickr30k_sg_output_64
│   │   ├── 36979.npz
│   │   ├── ...
│   │   └── 8251604257.npz
│   ├── flickr30k_graph_mask_1000_rm_duplicate
│   │   ├── 36979.npz
│   │   ├── ...
│   │   └── 8251604257.npz
│   ├── flickr30k_gt_graph_mask
│   │   ├── 36979.npz
│   │   ├── ...
│   │   └── 8251604257.npz
│   ├── cocotalk_label.h5
│   ├── cocotalk.json
│   ├── flickr30ktalk_label.h5
│   ├── flickr30ktalk.json
│   ├── flickr30k_img_wh.npy
│   ├── glove.6B.300d.pt
│   ├── gvd_all_dict.npy
│   └── sct_dict_test_grouped_gt_box.npy
│
├── misc
│   ├── consensus_reranking
│   │   ├── image_features_mRNN
│   │   │   ├── res_feat_101_dct_flickr30k.npy
│   │   │   └── res_feat_101_dct_mscoco_2014.npy
│   │   └── mscoco_anno_files
│   │       ├── anno_list_mscoco_trainModelVal_m_RNN.npy
│   │       ├── flickr30k_karpathy_train_val_anno_list.npy
│   │       └── karpathy_train_val_anno_list.npy
│   └── grounding
│       └── flickr30k_cleaned_class.json
│
└── pretrained
    ├── full_gc
    │   ├── infos_topdown-33000.pkl
    │   └── model-33000.pth
    ├── sub_gc_karpathy
    │   ├── infos_topdown-60000.pkl
    │   └── model-60000.pth
    ├── sub_gc_MRNN
    │   ├── infos_topdown-60000.pkl
    │   └── model-60000.pth
    ├── sub_gc_flickr
    │   ├── infos_topdown-16000.pkl
    │   └── model-16000.pth
    └── sub_gc_sup_flickr
        ├── infos_topdown-16000.pkl
        └── model-16000.pth
```
Finally, move the files to the same directories (`data`, `misc`, `pretrained`) in this repository. Please make sure that the final folder structure is kept the same.

The folder [`data`](https://github.com/YiwuZhong/Sub-GC/tree/master/data) contains the data that is derived from COCO Caption dataset and Flickr30K dataset. The folder [`pretrained`](https://github.com/YiwuZhong/Sub-GC/tree/master/pretrained) includes the pre-trained models of our paper. The folder `misc` includes the code and the data that are used for evaluation. Files of `res_feat*` are the global image features (ResNet-101) for each image in the datasets.

To download the code and models for SPICE evaluation, run
```
cd misc/coco-caption
bash get_stanford_models.sh
cp -r pycocoevalcap/spice/lib ../consensus_reranking/external/coco-caption/pycocoevalcap/spice/
```

Download [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) for grounding evaluation and place the uncompressed folder `stanford-corenlp-full-2018-02-27` under the `misc/grounding/tools` directory.
