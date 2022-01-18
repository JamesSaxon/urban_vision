#!/bin/bash 

for r in intx 1l 1w; do
for v in v3 1w 1l intx; do
for s in 608 416 320; do 

mkdir -p ml_img/${r}/TEST/yolo/${v}/${s}/

./evaluate_yolo.py --yolo_path ../yolo/${v}/ \
                   --yolo_size ${s} \
                   --input     ml_img/${r}/TEST/JPEGImages/ \
                   --output    ml_img/${r}/TEST/yolo/${v}/${s}/

done; done; done

 

for r in intx 1l 1w; do
for v in 1w 1l intx; do
for s in 0 1; do 

./evaluate_tpu.py --model  ../data/efdet_models/${v}-efdet-${s}_edgetpu.tflite \
                  --labels ../data/efdet_models/vehicles.txt \
                  --input  ml_img/${r}/TEST/JPEGImages/ \
                  --output ml_img/${r}/TEST/efdet/${v}/${s}/

done; done; done


