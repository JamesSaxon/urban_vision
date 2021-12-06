#!/home/jsaxon/.conda/envs/tf-25/bin/python

import numpy as np
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--input_tag", type = str, required = True, help = 'Path of the image files')
parser.add_argument("-n", "--efdet",     type = int, default = 0, choices = [0, 1, 2, 3], help="which efdet model")
parser.add_argument("-d", "--basedir",   type = str, default = "/net/scratch/jsaxon/img")
args = parser.parse_args()

tag = args.input_tag
efdet_level = args.efdet
base = args.basedir

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data, validation_data, test_data = object_detector.DataLoader.from_csv(f"{base}/{tag}/ml.csv")

if efdet_level == 0: spec = object_detector.EfficientDetLite0Spec(model_dir = f"{base}/{tag}/")
if efdet_level == 1: spec = object_detector.EfficientDetLite1Spec(model_dir = f"{base}/{tag}/")
if efdet_level == 2: spec = object_detector.EfficientDetLite2Spec(model_dir = f"{base}/{tag}/")
if efdet_level == 3: spec = object_detector.EfficientDetLite3Spec(model_dir = f"{base}/{tag}/")

model = object_detector.create(train_data        = train_data, 
                               validation_data   = validation_data, 
                               model_spec        = spec, 
                               epochs            = 70, 
                               batch_size        = 8, 
                               do_train          = True,
                               train_whole_model = True)

print(model.evaluate(test_data, batch_size = 8))

tflite_file = f'{tag}-efdet-{efdet_level}.tflite'
label_file  = f'{tag}.txt'
export_dir  = f'{base}/{tag}/'

model.export(tflite_filename = tflite_file, label_filename  = label_file,
             export_dir = export_dir, export_format = [ExportFormat.TFLITE, ExportFormat.LABEL])

print(model.evaluate_tflite(f'{base}/{tag}/{tag}-efdet-{efdet_level}.tflite', test_data))


