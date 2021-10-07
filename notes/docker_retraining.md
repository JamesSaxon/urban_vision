https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_detection_qat_tf1.ipynb
https://coral.ai/docs/edgetpu/retrain-detection/#start-training
https://coral.ai/docs/edgetpu/compiler/#performance-considerations	

## Get the files / repos

```
git clone https://github.com/tensorflow/models.git
cd models && git checkout f788046ca876a8820e05b0b48c1fc2e16b0955bc
cd ..

git clone https://github.com/google-coral/tutorials.git
cp -r tutorials/docker/object_detection/scripts/* models/research/
```

## 

Run with port 6006 and with root access, with or without gpus, at this stage (--gpus=all

```
sudo docker run -it -v $(pwd)/models/:/models --privileged -p 6006:6006 --name tf-train-gpu tensorflow/tensorflow:1.15.5-gpu-py3-jupyter /bin/bash
```

Then, in the container: 

```
apt update
apt install wget vim git
```

Note the hash.  Now restart and re-enger that container without root privileges

```
sudo docker start 3a39
sudo docker exec -it -u $(id -u):$(id -g) 3a39 /bin/bash
```

```
export PYTHONPATH=/models/research:/models/research/slim
```

Now get the files to base the SSD v1 retraining on:

```
./prepare_checkpoint_and_dataset.sh --network_type mobilenet_v1_ssd --train_whole_model false
```

Then adapt the data diretories (training files), structure, `label_map` file, esp. in pipeline.config, constants.sh

```
cd /models/research/
# N.B., if restarting training, remove past training directory, e.g.,
# rm -rf learn_cars/train/
./retrain_detection_model.sh --num_training_steps 1000 --num_eval_steps 200
```

### Watch it via 

```
tensorboard --logdir /models/research/learn_cars/train &
```

(Give it time to start, before exiting the container.)

### Convert models

```
ls -ltr learn_ped/train/model.ckpt-*index
./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 111

edgetpu_compiler --min_runtime_version 13 ~/proj/chalk/models/research/learn_ped/models/output_tflite_graph.tflite
```

