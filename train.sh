#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1

MODEL_TYPE=$1

if [ $MODEL_TYPE == "Sub_GC_MRNN" ]
then
  echo "Using Sub-GC model on COCO (MRNN split)"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 4000 --max_epochs 35 \
   --checkpoint_path logs/sub_gc_MRNN \
   --use_MRNN_split
fi

if [ $MODEL_TYPE == "Sub_GC_Kar" ]
then
  echo "Using Sub-GC model on COCO (Karpathy split)"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 4000 --max_epochs 35\
   --checkpoint_path logs/sub_gc_karpathy
fi

if [ $MODEL_TYPE == "Full_GC_Kar" ]
then
  echo "Using Full-GC model on COCO (Karpathy split)"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 3000 --max_epochs 35 \
   --checkpoint_path logs/full_gc   
fi

if [ $MODEL_TYPE == "Sub_GC_Flickr" ]
then
  echo "Using Sub-GC model on Flickr-30K"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 4000 --max_epochs 36 \
   --checkpoint_path logs/sub_gc_flickr \
   --input_label_h5 data/flickr30ktalk_label.h5 --input_json data/flickr30ktalk.json
fi

if [ $MODEL_TYPE == "Sub_GC_Sup_Flickr" ]
then
  echo "Using Sub-GC (Sup.) model on Flickr-30K"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 4000 --max_epochs 36 \
   --checkpoint_path logs/sub_gc_sup_flickr \
   --input_label_h5 data/flickr30ktalk_label.h5 --input_json data/flickr30ktalk.json --use_gt_subg
fi