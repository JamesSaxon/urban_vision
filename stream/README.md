# Streaming Tracker

The files in this directory 
  wrap detection libraries
  and implement tracking, using opencv.
The implementation is multithreaded, with 
  reading, writing, detection, and tracking,
  on four separate threads.
In addition, the core detection algorithms
  run on a CORAL (Google) `edgetpu`
  or on a local graphics card (for me, a `Quadro T1000`), 
  so these are effectively separated out too, to the coprocessor.

The options should be self-explaining; run `./stream.py -h`.
Settings can (and should!) be specified via a config file, 
  of which several examples can be found in the `conf/` directory.


