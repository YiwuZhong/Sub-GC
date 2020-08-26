#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1

MODEL_TYPE=$1

if [ $MODEL_TYPE == "Sub_GC_MRNN" ]
then
  echo "Using COCO (MRNN split) with Sub-GC model"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 4000 --max_epochs 35 \
   --checkpoint_path logs/sub_gc_MRNN \
   --use_MRNN_split
fi

if [ $MODEL_TYPE == "Sub_GC_Kar" ]
then
  echo "Using COCO (Karpathy split) with Sub-GC model"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 4000 --max_epochs 35\
   --checkpoint_path logs/sub_gc_karpathy
fi

if [ $MODEL_TYPE == "Full_GC_Kar" ]
then
  echo "Using COCO (Karpathy split) with Full-GC model"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 3000 --max_epochs 35 \
   --checkpoint_path logs/full_gc   
fi