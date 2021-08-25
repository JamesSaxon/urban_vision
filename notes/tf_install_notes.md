conda create --name tf pip python=3.8 jupyter

# The tensorflow 2.3 release does not successfully convert models, see here:

https://github.com/tensorflow/models/issues/9033
https://github.com/sayakpaul/E2E-Object-Detection-in-TFLite/blob/master/Object_Detection_in_TFLite.ipynb

conda create --name tf pip python=3.8 tqdm matplotlib jupyter 

source activate tf

python -m pip install --upgrade pip

## careful!! not robust to reinstalling, without cloning again!
git clone https://github.com/tensorflow/models.git
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

export TFMODELS=/media/jsaxon/brobdingnag/projects/urban_vision/models
export PYTHONPATH=$PYTHONPATH:$TFMODELS:$TFMODELS/research:$TFMODELS/research/slim

cp -av centernet_hg104_512x512_coco17_tpu-8/checkpoint models/research/object_detection/test_data/

## cv2 imshow does not work by default...
conda remove opencv
pip install opencv-contrib-python




######
###### Now try for the nightly, to be able to convert

conda create --name tfN pip python=3.8 tqdm matplotlib jupyter 

source activate tfN

python -m pip install --upgrade pip
pip install -q tf-nightly
python -c "import tensorflow as tf; print(tf.__version__)"


## careful!! not robust to reinstalling, without cloning again!
git clone https://github.com/tensorflow/models.git modelsN
cd modelsN/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# but that ruins the nightly build....
python -c "import tensorflow as tf; print(tf.__version__)"

# so check where the packages live, and delete AND uninstall BOTH
# http://github.com/tensorflow/tensorflow/issues/20778 
pip uninstall tensorflow
pip uninstall tf-nightly
rm -rf /home/jsaxon/anaconda3/envs/tfN/lib/python3.8/site-packages/tf_nightly-2.4.0.dev20201001.dist-info/
rm -rf /home/jsaxon/anaconda3/envs/tfN/lib/python3.8/site-packages/tensorflow
pip install tf-nightly

# Now it should work.  This is excessively bad.

cd ../../
export TFMODELS=/media/jsaxon/brobdingnag/projects/urban_vision/modelsN
export PYTHONPATH=$PYTHONPATH:$TFMODELS:$TFMODELS/research:$TFMODELS/research/slim



python object_detection/export_tflite_graph_tf2.py --pipeline_config_path $MODEL/pipeline.config --trained_checkpoint_dir $MODEL/checkpoint/ --output_directory `pwd`

and then 

>>> import tensorflow as tf
>>> tf.__version__
>>> converter = tf.lite.TFLiteConverter.from_saved_model('/media/jsaxon/brobdingnag/projects/urban_vision/modelsN/research/saved_model/')
>>> converter.optimizations = [tf.lite.Optimize.DEFAULT]
>>> tflite_model = converter.convert()
>>> open('ssd.tflite', 'wb').write(tflite_model)

This _looks_ fine, until we realize, belatedly that 

https://github.com/google-coral/edgetpu/issues/210#issuecomment-683863816

TF2 is still not supported for edge devices.  Great waste of time.

### TF1

conda create --name tf1 pip python=3.6 tqdm matplotlib jupyter cython 

source activate tf1

# https://github.com/cocodataset/cocoapi/issues/94
git clone https://github.com/tensorflow/models.git models1
cd cocoapi/PythonAPI/
vi setup.py -->> add extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'] to Extension
make install 
cd ../../


python -m pip install --upgrade pip

git clone https://github.com/tensorflow/models.git models1
cd models1/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install --use-feature=2020-resolver .


pip install tensorflow==1.15

export TFMODELS=/media/jsaxon/brobdingnag/projects/urban_vision/models1
export PYTHONPATH=$PYTHONPATH:$TFMODELS:$TFMODELS/research:$TFMODELS/research/slim





