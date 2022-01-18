#!/bin/bash 

for d in 211031; do 
  for c in 106 108 109 120 213; do 
    for v in /home/jsaxon/proj/vid/city/new/vid/lf/${c}/${d}_*.mp4; do 

      ##  for s in 320 416 608; do 
      ##    ./stream.py -c conf/lf.conf -i ${v} --odir odata/lf/${c}/ --tag yolo${s}_lf --yolo_size ${s} 
      ##  done

      for m in 0 1; do 
        ./stream.py -c conf/lf.conf -i ${v} --odir odata/lf/${c}/ --tag efdet_lf${m} \
                    --model ../data/efdet_models/lf-efdet-${m}_edgetpu.tflite
      done 

    done
  done
done


