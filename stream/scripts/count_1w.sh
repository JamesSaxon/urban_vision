#!/bin/bash 

for d in 211030; do 
  for l in clark_grand; do # lower_wacker state_adams_e state_jackson state_monroe_w; do 
    for v in /home/jsaxon/proj/vid/city/new/vid/1w/${l}/${d}_[12][480]*.mp4; do 
      ./stream.py -c conf/1w.conf -i ${v} --odir odata/1w/${l}/ --tag yolo608_1w --yolo_size 608
      ./stream.py -c conf/1w.conf -i ${v} --odir odata/1w/${l}/ --tag efdet1_1w
    done
  done
done


