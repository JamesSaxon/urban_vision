# Urban Vision: Tracking with Computer Vision

The project of recording human activity in urban environments entails three parts:
1. Getting data
2. Training or finding a detector
3. Tracking detected objects over time.
This repository focuses primarily on the third problem, though there are a few hints at the second -- how to retrain single shot detectors.

The directory of interest is probably `stream/`, which contains two classes of interest -- `detector.py` and `tracker.py` -- which are controlled by the eponymous `stream.py` script, which parallelizes tasks.

The detector can use either YOLO or SSD models;
  these require `edgetpu` and `CUDA`/GPU libraries,
  which is not completely trivial, and which is not addressed here.



