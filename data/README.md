# Data

Please follow the instruction in the `Data` section of the mainpage to download the data files. The file details are as follows:

* `COCO_sg_output_64/` & `flickr30k_sg_output_64/`: contain object detection features from [Bottom-Up](https://github.com/peteanderson80/bottom-up-attention) as well as the detected scene graphs from [Motif-Net](https://github.com/rowanz/neural-motifs) pre-trained on Visual Genome. 
* `COCO_graph_mask_1000_rm_duplicate/` & `flickr30k_graph_mask_1000_rm_duplicate/`: store the sampled sub-graphs.
* `sct_dict_test_grouped_gt_box.npy`: include the ground-truth bounding boxes (grounding annotation) from [controllability evaluation](https://github.com/aimagelab/show-control-and-tell).
* `flickr30k_gt_graph_mask/`: store the sub-graphs that are extracted by the ground-truth bounding boxes and will be used for controllability experiment. 
* `gvd_all_dict.npy`: is a dictionary from [grounding evaluation](https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch), which contains the mapping between the detection categories and caption words, and will be used for collecting the grounding results.
* `glove.6B.300d.pt` contains the GloVe word embeddings.
* The rest of the files are dataset-related annotations.