#!/bin/bash 

for d in 211031; do 
  for l in 1 2 4 5 6; do 
    for v in /home/jsaxon/proj/vid/city/new/vid/1l/*_NB${l}/${d}_*.mp4; do 
    # ./stream.py -c conf/1l.conf -i ${v} --odir odata/1l/90_33st_NB${l}/ --tag yolo_1l --yolo_size 608
      ./stream.py -c conf/1l.conf -i ${v} --odir odata/1l/90_33st_NB${l}/ --tag efdet0_1l
    done
  done
done


