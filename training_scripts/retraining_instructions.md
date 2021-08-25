Instructions for running retraining in docker container

1. Start a new container:
docker run --name edgetpu-detect-lsd -it --privileged -p 6006:6006 --mount type=bind,src=/media/jsaxon/brobdingnag/projects/edgetpu/adetection/retrain_lsd,dst=/tensorflow/models/research/learn_lsd detect-tutorial
  * Go to /media/jsaxon/brobdingnag/projects/edgetpu

2. In the docker container, edit /tensorflow/models/research/constants.sh:
line 24: LEARN_DIR="${OBJ_DET_DIR}/learn_hwp"
line 25: DATASET_DIR="${LEARN_DIR}"

3. Check that /tensorflow/models/research/learn/ directory contains training and validation records in `data/`,
   a `ckpt-v?` directory with pipeline.config and checkpoint files, and the file `label_map.pbtxt` referenced in the pipeline..
   Also make sure the train directory doesn't exist.
   * You should already have copied over the converted tfrecord files to `learn/data`
   * You also need to either edit label_map.pbtxt (of the form below) or change the pipeline config in the ckpt directories.
   * Inside `learn_*` get the ckpt files from pets or another run, or wget, gunzip, and untar one of these models (v1 or v2), and then move it to `ckpt`
     * https://coral.ai/models/#object-detection
   * Within the pipeline file, respecify:
     * the number of classes
     * the glob tags for the training and validation files
     * the mscoco → your new label map, etc. --
   * Alternatively, follow the instructions here:
     * https://coral.ai/docs/edgetpu/retrain-detection/#configure-your-training-pipeline
     * Note that for `fine_tune_checkpoint`, the model checkpoint as unpacked is `ckpt-v1/model.ckpt`, even though the actual paths are different.

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

4. In the /tensorflow/models/research/ directory, run
   ```
   ./retrain_detection_model.sh --num_training_steps 2500 --num_eval_steps 500
   ```
   We've used eval = training / 5.



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
