#!/usr/bin/env python 

import sys

import cv2
import tensorflow as tf
import numpy as np
import argparse

import train

alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, help='Path of the tfrecord file.')
parser.add_argument("--select_roi", default = False, action = "store_true")
parser.add_argument('--ymin', help='Lower y bound (frac)', type = float, default=0.406)
parser.add_argument('--ymax', help='Upper y bound (frac)', type = float, default=0.95)
parser.add_argument('--xmin', help='Lower x bound (frac)', type = float, default=0.155)
parser.add_argument('--xmax', help='Upper x bound (frac)', type = float, default=0.838)
parser.add_argument('--frac', help='Threshold for the fraction of object to include',
                    default=0.25, type=float)
parser.add_argument("--xred", default = 1, type = float)
parser.add_argument("--yred", default = 1, type = float)
parser.add_argument("--plim", default = 100, type = int,
                    help='Lower limit on pixel area in a 300x300 image')

args = parser.parse_args()

file_path = args.file
file_name = file_path.split("/")[-1]


#Set output path by inserting "roi" into the file name.
file_path_list = file_path.split('_')
file_path_list.insert(-1, 'roi')
if args.xred > 1: file_path_list.insert(-1, 'xred{:.2f}'.format(args.xred))
if args.yred > 1: file_path_list.insert(-1, 'yred{:.2f}'.format(args.yred))
output_file = '_'.join(file_path_list)

writer = tf.io.TFRecordWriter(output_file)


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

def get_image_dims(ds):

    for example in ds:
        image_raw = example['image/encoded'].numpy()
        image_encoded = np.frombuffer(image_raw, np.uint8)
        img = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)

        return img.shape[1], img.shape[0]

tfrecord_dataset = tf.data.TFRecordDataset([file_path])
parsed_dataset = tfrecord_dataset.map(train.read_tfrecord)

image_width, image_height = get_image_dims(parsed_dataset)

#ROI list from grid.
if args.select_roi:

    for exi, example in enumerate(parsed_dataset):

        if exi < 40: continue

        image_raw = example['image/encoded'].numpy()
        image_encoded = np.frombuffer(image_raw, np.uint8)
        img = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)
        break

    print("frame size", image_width, image_height)

    img = cv2.resize(img, None, fx = 1 / 2, fy = 1 / 2)

    ROI = cv2.selectROI(img)
    cv2.destroyWindow("ROI selector")

    XMIN, XMAX = 2 * ROI[0], 2 * (ROI[0] + ROI[2])
    YMIN, YMAX = 2 * ROI[1], 2 * (ROI[1] + ROI[3])

    print("You just selected ROI: --xmin {:.02f} --xmax {:.02f} --ymin {:.02f} --ymax {:.02f}"\
          .format(XMIN / image_width, XMAX / image_width, YMIN / image_height, YMAX / image_height))

    with open("extract_roi.sh", "a") as out:
        out.write("/media/jsaxon/brobdingnag/projects/urban_vision/sliding_roi.py --xmin {:.02f} --xmax {:.02f} --ymin {:.02f} --ymax {:.02f} --xred {:.02f} --yred {:.02f} --file {:s}"\
                  .format(XMIN / image_width, XMAX / image_width, YMIN / image_height, YMAX / image_height,
                          args.xred, args.yred, args.file))

    sys.exit()

else:

    XMIN = args.xmin * image_width
    XMAX = args.xmax * image_width 
    YMIN = args.ymin * image_height
    YMAX = args.ymax * image_height


# These are single video files with fixed dimensions.
roi_width  = int(round((XMAX - XMIN) / args.xred))
roi_height = int(round((YMAX - YMIN) / args.yred))
image_size = roi_width * roi_height

x_start = XMIN
y_start = YMIN
x_slide = (XMAX - XMIN) - roi_width 
y_slide = (YMAX - YMIN) - roi_height


for exi, example in enumerate(parsed_dataset):

    #Extract image, slice, and re-encode.
    image_raw = example['image/encoded'].numpy()
    image_encoded = np.frombuffer(image_raw, np.uint8)
    image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)

    # So this will be a random offset for each image.
    xs, ys = np.random.random(2)
    roi_xmin = int(round(x_start + xs * x_slide))
    roi_ymin = int(round(y_start + ys * y_slide))
    roi_xmax = roi_xmin + roi_width
    roi_ymax = roi_ymin + roi_height
    
    new_image = image[roi_ymin:roi_ymax,roi_xmin:roi_xmax,:]

    _, im_buf_arr = cv2.imencode(".jpg", new_image)

    new_source = str(example['image/source_id'].numpy(), 'utf-8')
    if args.xred > 1: new_source += "_xs{:.02f}".format(xs)
    if args.yred > 1: new_source += "_ys{:.02f}".format(ys)

    #Initialize dictionary for new tfrecord.
    output_dict = {'video': str(example['image/video'].numpy(), 'utf-8'),
                   'source_id': new_source,
                   'image_width': roi_width,
                   'image_height': roi_height,
                   'image_encoded':im_buf_arr.tobytes(),
                   'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[],
                   'classes_text':[]}

    #Iterate through objects, check if object is inside ROI, if so, add to dictionary.
    labels = example['image/object/class/text'].numpy().astype(str)
    for i, label in enumerate(labels):
        class_num = int(example['image/object/class/label'].numpy().astype(str)[i])
        xmin = int(example['image/object/bbox/xmin'].numpy().astype(float)[i]*image_width)
        xmax = int(example['image/object/bbox/xmax'].numpy().astype(float)[i]*image_width)
        ymin = int(example['image/object/bbox/ymin'].numpy().astype(float)[i]*image_height)
        ymax = int(example['image/object/bbox/ymax'].numpy().astype(float)[i]*image_height)

        if xmin > roi_xmax: continue
        if xmax < roi_xmin: continue
        if ymin > roi_ymax: continue
        if ymax < roi_ymin: continue

        box_width   = xmax - xmin
        box_height  = ymax - ymin
        in_box_area = box_width * box_height

        new_xmin = max(0, xmin - roi_xmin)
        new_ymin = max(0, ymin - roi_ymin)
        new_xmax = min(roi_width,  xmax - roi_xmin)
        new_ymax = min(roi_height, ymax - roi_ymin)

        # Write it if, in the rescaled image space, of 300x300, it exceeds plim pixels.
        out_box_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)

        if out_box_area < args.frac * in_box_area: continue
        if (out_box_area / image_size) < (args.plim / 300 / 300): continue

        if not((0 <= new_xmin <= roi_width)  and \
               (0 <= new_xmax <= roi_width)  and \
               (0 <= new_ymin <= roi_height) and \
               (0 <= new_ymax <= roi_height)):

            print("   width={}                 height={}".format(roi_width, roi_height))
            print("roi_xmin:{}  roi_xmax:{}  roi_ymin:{}  roi_ymax:{}  ".format(roi_xmin, roi_xmax, roi_ymin, roi_ymax))
            print("    xmin:{}      xmax:{}      ymin:{}      ymax:{}  ".format(xmin, xmax, ymin, ymax))
            print("new_xmin:{}  new_xmax:{}  new_ymin:{}  new_ymax:{}  ".format(new_xmin, new_xmax, new_ymin, new_ymax))

            print(output_dict["image_width"], output_dict["image_height"], roi_width, roi_height)
            sys.exit()

        train.update_dict_coords(output_dict, class_num, label, (new_xmin, new_ymin), (new_xmax,new_ymax))

    if output_dict["classes"]:
        tf_example = train.create_tf_example(output_dict)
        writer.write(tf_example.SerializeToString())

print("Wrote {} records.".format(exi))

writer.close()
