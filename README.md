# Sub-GC

This repository includes the Pytorch code for our paper "Comprehensive Image Captioning via Scene Graph Decomposition" in ECCV 2020.

![](model_overview.png)

[[Project Page]](http://pages.cs.wisc.edu/~yiwuzhong/Sub-GC.html) [[Paper]](https://arxiv.org/pdf/2007.11731.pdf)

For now, we provide the code for training and testing on COCO Caption dataset. The code for Flickr30k will be released shortly. We will keep the repo updated. Stay tuned!

## Dependencies
* Python 2.7
* Pytorch 1.3.0+

Python and Pytorch can be installed by anaconda, run
```
conda create --name ENV_NAME python=2.7
source activate ENV_NAME
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
where `ENV_NAME` and [cudatoolkit version](https://pytorch.org/) can be specified by your own.

For the other dependencies, run `pip install -r requirements.txt` to install.

## Data

First download all the data from [Google Drive](https://drive.google.com/drive/folders/1mCx8R8d36ZpUSoVZKExs0FDA_IXiAiZA?usp=sharing). After that, extract the files from the compressed `*.zip` files and merge folders of `*_part1/`, `*_part2/` and `*_part3/` into a new folder `COCO_sg_output_64/`. The folder structure should be as follows:
```
├── data
│   ├── COCO_graph_mask_1000_rm_duplicate
│   │   ├── 9.npz
│   │   ├── ...
│   │   └── 581929.npz
│   ├── COCO_sg_output_64
│   │   ├── 9.npz
│   │   ├── ...
│   │   └── 581929.npz
│   ├── cocotalk_label.h5
│   ├── cocotalk.json
│   ├── flickr30ktalk_label.h5
│   ├── flickr30ktalk.json
│   ├── glove.6B.300d.pt
│   └── gvd_all_dict.npy
│
├── misc
│   └── consensus_reranking
│       ├── image_features_mRNN
│       │   ├── res_feat_101_dct_flickr30k.npy
│       │   └── res_feat_101_dct_mscoco_2014.npy
│       └── mscoco_anno_files
│           ├── anno_list_mscoco_trainModelVal_m_RNN.npy
│           ├── flickr30k_karpathy_train_val_anno_list.npy
│           └── karpathy_train_val_anno_list.npy
└── pretrained
    ├── full_gc
    │   ├── infos_topdown-33000.pkl
    │   └── model-33000.pth
    ├── sub_gc_karpathy
    │   ├── infos_topdown-60000.pkl
    │   └── model-60000.pth
    └── sub_gc_MRNN
        ├── infos_topdown-60000.pkl
        └── model-60000.pth
```
Finally, move the files to the same directories (`data`, `misc`, `pretrained`) in this repository. Please make sure that the final folder structure is kept the same.

The sub-folder `data/COCO_sg_output_64/` contains object detection features from [Bottom-Up](https://github.com/peteanderson80/bottom-up-attention) as well as the detected scene graphs from [Motif-Net](https://github.com/rowanz/neural-motifs) pre-trained on Visual Genome. The sub-folder `data/COCO_graph_mask_1000_rm_duplicate/` stores the sampled sub-graphs. The folder `pretrained/` includes the pre-trained models of our paper. The file `glove.6B.300d.pt` contains the GloVe word embeddings. Files of `res_feat*` are the global image features (ResNet-101) for each image in the datasets. The rest of the files are dataset-related annotations.

To download the code and models for SPICE, run
```
cd coco-caption
bash get_stanford_models.sh
cp -r pycocoevalcap/spice/lib ../misc/consensus_reranking/external/coco-caption/pycocoevalcap/spice/
```

## Model Training

To train our image captioning models, run the script
```
bash train.sh MODEL_TYPE
```
by replacing `MODEL_TYPE` with one of `[Sub_GC_MRNN, Sub_GC_Kar, Full_GC_Kar]`. `MODEL_TYPE` specifies the dataset, the data split and the model used for training. See details below.

* `Sub_GC_MRNN`: train a sub-graph captioning model on M-RNN split of COCO (Table 2 in our paper)
* `Sub_GC_Kar`: train a sub-graph captioning model on Karpathy split of COCO (Table 3 in our paper)
* `Full_GC_Kar`: train a full-graph captioning model on Karpathy split of COCO (Table 3 in our paper)

You can set `CUDA_VISIBLE_DEVICES` in `train.sh` to specify which GPUs are used for model training (e.g., the default script uses 2 GPUs).

## Model Evaluation
The evaluation is divided into 2 steps
- The trained model is first used to generate captions
- The generated sentences are evaluated in terms of diversity, top-1 accuracy, grounding and controllability.

### Caption Generation

To generate captions, run the script
```
bash test.sh MODEL_TYPE
```
by replacing `MODEL_TYPE` with one of `[Sub_GC_MRNN, Sub_GC_S_MRNN, Sub_GC_Kar, Full_GC_Kar]`. `MODEL_TYPE` specifies the dataset, the data split and the model used for sentence generation. See details below.

* `Sub_GC_MRNN`: use the sub-graph captioning model Sub-GC on M-RNN split of COCO (Table 2 in our paper)
* `Sub_GC_S_MRNN`: use Sub-GC with top-k sampling (Sub-GC-S) on M-RNN split of COCO (Table 2 in our paper)
* `Sub_GC_Kar`: use the sub-graph captioning model Sub-GC on Karpathy split of COCO (Table 3 in our paper)
* `Full_GC_Kar`: use the full graph captioning model Full-GC on Karpathy split of COCO (Table 3 in our paper)

The inference results will be saved in a `*.npy` file at the same folder as the model checkpoint (e.g., `pretrained/sub_gc_MRNN`). `$CAPTION_FILE` will be used as the name of generated `*.npy` file in the following instructions.

### Diversity Evaluation

Move the generated `$CAPTION_FILE` into folder `misc/diversity` and run
```
cd misc/diversity
python diversity_score.py --input_file $CAPTION_FILE
```
To evaluate the metric of mBLEU-4 (takes much longer time than other metrics), run
```
cd misc/diversity
python diversity_score.py --input_file $CAPTION_FILE --evaluate_mB4
```

### Top-1 Accuracy Evaluation

In our paper, we report the top-1 accuracy of the best caption selected by sGPN+consensus. To reproduce the results, move the generated `$CAPTION_FILE` into folder `misc/consensus_reranking/hypotheses_mRNN` and run:
```
cd misc/consensus_reranking
python cr_mRNN_demo.py --top_k 4 --caption_file $CAPTION_FILE --dataset coco --split MRNN
```
This will apply consensus reranking on the top 4 captions selected by our sGPN scores as described in our paper. The arguments of `--dataset` and `--split` specify the dataset (`coco` or `flickr30k`) and the split (`MRNN` or `karpathy`), respectively. Note that only COCO dataset is supported by current code.

If you want to evaluate the top-1 caption selected by our sGPN or the top-1 accuracy for Full-GC, set `--only_sent_eval` to `1` in `test.sh` and rerun the bash file. If you want to evaluate the oracle scores which will take a few hours, set `--only_sent_eval` to `1` and add `--orcle_num 1000` in `test.sh`, and rerun the bash file.

### Grounding Evaluation (on Flickr30k)
Under construction.

### Controllability Evaluation (on Flickr30k)
Under construction.

## Acknowledgement

This repository was built based on [Ruotian Luo's implementation](https://github.com/ruotianluo/self-critical.pytorch/tree/2.5) for image captioning and [Graph-RCNN](https://github.com/jwyang/graph-rcnn.pytorch). Partial evaluation protocols were implemented based on several code repositories, including: [coco-caption](https://github.com/tylin/coco-caption), [consensus reranking](https://github.com/mjhucla/mRNN-CR), [grounding evaluation](https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch), and [controllability evaluation](https://github.com/aimagelab/show-control-and-tell).

## Reference
If you are using our code, please consider citing our paper.
```
@inproceedings{zhong2020comprehensive,
  title={Comprehensive Image Captioning via Scene Graph Decomposition},
  author={Zhong, Yiwu and Wang, Liwei and Chen, Jianshu and Yu, Dong and Li, Yin},
  booktitle={ECCV},
  year={2020}
}
```
