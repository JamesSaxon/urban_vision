import argparse
import platform
import subprocess

import cv2

import numpy as np

from PIL import Image
from PIL import ImageDraw

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from glob import glob
from tqdm import tqdm

import sys

colors = ["red", "yellow", "green"]

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True)
  parser.add_argument('--label', help='Path of the labels file.')
  parser.add_argument('--input', help = 'Directory of input images.', required=True)
  parser.add_argument('-k', default = 10, type = int)
  parser.add_argument('--thresh', default = 0.5, type = float)
  parser.add_argument("--categs", default = [], nargs = "+")
  parser.add_argument("--frames", default = 0, type = int)

  parser.add_argument('--keep_aspect_ratio', default = False, dest='keep_aspect_ratio', action='store_true',)

  args = parser.parse_args()

  # Initialize engine.
  engine = DetectionEngine(args.model)
  labels = dataset_utils.read_label_file(args.label) if args.label else None

  vid = cv2.VideoCapture(args.input)
  
  out = cv2.VideoWriter(args.input.replace("mov", "mp4").replace(".mp4", "_det.mp4"), 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (int(vid.get(3)), int(vid.get(4))))

  nframe = 0
  while True:

      ret, frame = vid.read()

      nframe += 1
      print(nframe)
      if not ret: break
      if args.frames and nframe > args.frames: break

      img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

      draw = ImageDraw.Draw(img)

      # Run inference.
      ans = engine.detect_with_image(img, threshold = args.thresh, keep_aspect_ratio=args.keep_aspect_ratio, relative_coord=False, top_k = args.k)

      # Save result.
      print('=========================================')

      if ans:
          for obj in ans:


              label = labels[obj.label_id]
              if len(args.categs) and label not in args.categs: continue

              if labels:
                  print(labels[obj.label_id] + ",", end = " ")

              print('conf. = ', obj.score)


              # Draw a rectangle.
              box = obj.bounding_box.flatten().tolist()
              draw.rectangle(box, outline = colors[args.categs.index(label)] if len(args.categs) else "red")
              # print('box = ', box)
              print('-----------------------------------------')

      else: print('No object detected!')

      out.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
      print('Recorded frame {}'.format(nframe))

  out.release()


if __name__ == '__main__': main()



