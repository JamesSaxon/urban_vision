Instructions for running retraining in docker container

1. Start a new container:
docker run --name edgetpu-detect-lsd -it --privileged -p 6006:6006 --mount type=bind,src=/media/jsaxon/brobdingnag/projects/edgetpu/adetection/retrain_lsd,dst=/tensorflow/models/research/learn_lsd detect-tutorial
  * Go to /media/jsaxon/brobdingnag/projects/edgetpu

2. In the docker container, edit /tensorflow/models/research/constants.sh:
line 24: LEARN_DIR="${OBJ_DET_DIR}/learn_hwp"
line 25: DATASET_DIR="${LEARN_DIR}"

3. Check that /tensorflow/models/research/learn_hwp directory contains training and validation records,
    a ckpt directory with pipeline.config and checkpoint files, and the file lsd_label_map.pbtxt.
    Also make sure the train directory doesn't exist.
  * You should already have copied over the converted tfrecord files.
  * You also need a cat hwp_label_map.pbtxt, of the form below
  * Inside learn_hwp (or whatever) can wget, gunzip, and untar one of these models (v1 or v2), and then move it to `ckpt`
    * https://coral.ai/models/#object-detection
  * Within the pipeline file, respecify the number of classes, the mscoco → your new label map, etc. -- follow the instructions here:
    * https://coral.ai/docs/edgetpu/retrain-detection/#configure-your-training-pipeline
    * Note that for `fine_tune_checkpoint`, the model checkpoint as unpacked is `ckpt/model.ckpt`, even though the actual paths are different.

```
item {
  id: 1
  name: 'car'
}

item {
  id: 2
  name: 'bus'
}
```

4. In the /tensorflow/models/research/ directory, set training parameters:
    `NUM_TRAINING_STEPS=500 && NUM_EVAL_STEPS=100`

5. In the same directory:
    `./retrain_detection_model.sh --num_training_steps ${NUM_TRAINING_STEPS} --num_eval_steps ${NUM_EVAL_STEPS}`



Changes to retraining scripts:

error: File "/tensorflow/models/research/slim/nets/inception_resnet_v2.py", line 375, in <module>
    batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
AttributeError: 'module' object has no attribute 'v1'

changes:
in /tensorflow/models/research/slim/nets/inception_resnet_v2.py, line 375 - batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,



error: File "/tensorflow/models/research/slim/nets/inception_utils.py", line 39, in <module>
    batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
AttributeError: 'module' object has no attribute 'v1'

change: in /tensorflow/models/research/slim/nets/inception_utils.py, line 39 - batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,



error: File "/tensorflow/models/research/slim/nets/resnet_utils.py", line 231, in <module>
    batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS):
AttributeError: 'module' object has no attribute 'v1'

change: in /tensorflow/models/research/slim/nets/resnet_utils.py, line 231 - batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS)



error: File "/tensorflow/models/research/slim/nets/mobilenet_v1.py", line 438, in <module>
    batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
AttributeError: 'module' object has no attribute 'v1'

change: in /tensorflow/models/research/slim/nets/mobilenet_v1.py, line 438 - batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,



error: File "/tensorflow/models/research/slim/nets/mobilenet/mobilenet.py", line 399, in <module>
    def global_pool(input_tensor, pool_op=tf.compat.v2.nn.avg_pool2d):
AttributeError: 'module' object has no attribute 'v2'

change: /tensorflow/models/research/slim/nets/mobilenet/mobilenet.py, line 399 - def global_pool(input_tensor, pool_op=tf.nn.avg_pool):



error: File "/tensorflow/models/research/slim/nets/mobilenet_v1.py", line 469, in mobilenet_v1_arg_scope
    weights_init = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
AttributeError: 'module' object has no attribute 'v1'

change: /tensorflow/models/research/slim/nets/mobilenet_v1.py, line 469 - weights_init = tf.truncated_normal_initializer(stddev=stddev)



error: File "/tensorflow/models/research/slim/nets/mobilenet_v1.py", line 236, in mobilenet_v1_base
    with tf.compat.v1.variable_scope(scope, 'MobilenetV1', [inputs]):
AttributeError: 'module' object has no attribute 'v1'

change: /tensorflow/models/research/slim/nets/mobilenet_v1.py, line 236 - tf.variable_scope(scope, 'MobilenetV1', [inputs]):