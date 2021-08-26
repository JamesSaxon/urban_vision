#!/bin/bash 

vid=../../../data/cv/vid/burnham/55

for x in ${vid}/202009??_*.MOV
  do ./stream.py -c burnham.conf -i ${x}
done

