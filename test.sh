#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=$1

if [ $MODEL_TYPE == "Sub_GC_MRNN" ]
then
  echo "Using COCO (MRNN split) with Sub-GC model"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 1 \
   --gpn_nms_thres 0.55 --gpn_max_subg 1000 \
   --model pretrained/sub_gc_MRNN/model-60000.pth \
   --infos_path pretrained/sub_gc_MRNN/infos_topdown-60000.pkl \
   --only_sent_eval 0 \
   --use_MRNN_split
fi

if [ $MODEL_TYPE == "Sub_GC_S_MRNN" ]
then
  echo "Using COCO (MRNN split) with Sub-GC-S model"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 1 \
   --gpn_nms_thres 0.55 --gpn_max_subg 1000 \
   --model pretrained/sub_gc_MRNN/model-60000.pth \
   --infos_path pretrained/sub_gc_MRNN/infos_topdown-60000.pkl\
   --only_sent_eval 0 \
   --use_MRNN_split \
   --use_topk_sampling 1 --topk_temp 0.6 --the_k 3
fi

if [ $MODEL_TYPE == "Sub_GC_Kar" ]
then
  echo "Using COCO (Karpathy split) with Sub-GC model"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 2 \
   --gpn_nms_thres 0.75 --gpn_max_subg 1000 \
   --model pretrained/sub_gc_karpathy/model-60000.pth \
   --infos_path pretrained/sub_gc_karpathy/infos_topdown-60000.pkl\
   --only_sent_eval 0
fi

if [ $MODEL_TYPE == "Full_GC_Kar" ]
then
  echo "Using COCO (Karpathy split) with Full-GC model"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 3 \
   --model pretrained/full_gc/model-33000.pth \
   --infos_path pretrained/full_gc/infos_topdown-33000.pkl\
   --only_sent_eval 0 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi

