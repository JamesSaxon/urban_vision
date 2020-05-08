import cv2
import tensorflow as tf
import numpy as np
import argparse

import train

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, help='Path of the tfrecord file.')
parser.add_argument('--ymin', help='Lower y bound (frac)', type = float, default=0.203)
parser.add_argument('--ymax', help='Upper y bound (frac)', type = float, default=0.625)
parser.add_argument('--xmin', help='Lower x bound (frac)', type = float, default=0.0977)
parser.add_argument('--xmax', help='Upper x bound (frac)', type = float, default=0.781)
parser.add_argument('--frac', help='Threshold for the fraction of object to include',
                    default=0.2)

args = parser.parse_args()


file_path = args.file

#Set output path by inserting "roi" into the file name.
file_path_list = file_path.split('_')
file_path_list.insert(-1, 'roi')
output_path = '_'.join(file_path_list)
writer = tf.io.TFRecordWriter(output_path)


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

tfrecord_dataset = tf.data.TFRecordDataset([file_path])
parsed_dataset = tfrecord_dataset.map(train.read_tfrecord)

for example in parsed_dataset:
    #Extract image, slice, and re-encode.
    image_raw = example['image/encoded'].numpy()
    image_encoded = np.frombuffer(image_raw, np.uint8)
    image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)
    image_width = image.shape[1]
    image_height = image.shape[0]
    _xmin = int(args.xmin*image_width)
    _xmax = int(args.xmax*image_width)
    _ymin = int(args.ymin*image_height)
    _ymax = int(args.ymax*image_height)
    new_image = image[_ymin:_ymax,_xmin:_xmax,:]
    _, im_buf_arr = cv2.imencode(".jpg", new_image)

    #Initialize dictionary for new tfrecord.
    output_dict = {'video': str(example['image/video'].numpy(), 'utf-8'),
                   'source_id': str(example['image/source_id'].numpy(), 'utf-8'),
                   'image_width':new_image.shape[1],
                   'image_height':new_image.shape[0],
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

        box_width = xmax - xmin
        box_height = ymax - ymin
        xmax_lowest = _xmin + box_width*args.frac
        xmin_highest = _xmax - box_width*args.frac
        ymax_lowest = _ymin + box_height*args.frac
        ymin_highest = _ymax - box_height*args.frac

        if xmin < xmin_highest and xmax > xmax_lowest and ymin < ymin_highest and ymax > ymax_lowest:
            new_xmin = max(0, xmin - _xmin)
            new_xmax = min(_xmax - _xmin, xmax - _xmin)
            new_ymin = max(0, ymin - _ymin)
            new_ymax = min(_ymax - _ymin, ymax - _ymin)
            train.update_dict_coords(output_dict, class_num, label, (new_xmin, new_ymin), (new_xmax,new_ymax))

    tf_example = train.create_tf_example(output_dict)
    writer.write(tf_example.SerializeToString())
writer.close()
