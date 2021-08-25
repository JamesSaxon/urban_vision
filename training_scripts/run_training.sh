#!/bin/bash 

export UV=/media/jsaxon/brobdingnag/projects/urban_vision
export TFMODELS=$UV/models1
export PYTHONPATH=$PYTHONPATH:$TFMODELS:$TFMODELS/research:$TFMODELS/research/slim
export MODEL=ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18
export MODEL_DIR=$UV/training/$MODEL

cd $UV/models1/research/

python object_detection/model_main.py \
  --pipeline_config_path=$UV/training/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/pipeline.config \
  --model_dir=$UV/training/train \
  --num_train_steps=100 --num_eval_steps=500

