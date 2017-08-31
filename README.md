# Inception_v4_slim

 **模型** ：slim框架下的Inception_v4模型 

Inception_v4的Checkpoint：http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz 

 **数据集** ：google的flower数据集http://download.tensorflow.org/example_images/flower_photos.tgz 5种类别的花

本文内容是我学习智亮老师图像识别课程的一些笔记与想法，加深学习，并方便自己回顾。智亮老师的课程讲的还是挺不错的，受益匪浅。
 
 **代码** ：https://codeload.github.com/isiosia/models/zip/lession 

 **GitHub** ：https://github.com/isiosia/models/tree/lession

 **数据准备** 

数据集下下来之后按/home/lwp/data/flower/my_flower_5路径放好，可以看到它是这个样子的，每个类的花一个文件夹

![输入图片说明](http://img.blog.csdn.net/20170727103647557?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

打开一个我们可以看到里面是各种图片

![输入图片说明](http://img.blog.csdn.net/20170727103756273?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

在模型目录source/models/slim下有一个脚本文件convert_tfrecord.sh 
convert_tfrecord.sh文件内容如下：

```
source env_set.sh
python download_and_convert_data.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR
```

可以看到通过env_set.sh传递变量 
env_set.sh文件内容如下：

```
export DATASET_NAME=my_flower_5
export DATASET_DIR=/home/lwp/data/flower
export CHECKPOINT_PATH=/home/lwp/pre_trained/inception_v4.ckpt
export TRAIN_DIR=/tmp/my_train_20170725
```

文件定义了：

- DATASET_NAME：数据集名称
- DATASET_DIR：数据集路径
- CHECKPOINT_PATH：预训练的inception_v4模型路径
- TRAIN_DIR：训练生成checkpoint存储路径

环境变量配置完后进入到模型目录下

```
$ cd source/models/slim
```

执行脚本：

```
$ ./convert_tfrecord.sh
```

完成后数据就准备好了 

![输入图片说明](http://img.blog.csdn.net/20170727105532449?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")


 **预训练模型准备** 

Inception_v4的Checkpoint：http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz 
下载好之后存放在下面路径（路径在env_set.sh中定义）：

```
/home/lwp/pre_trained
```

 **运行训练脚本** 

（在修改好模型相关参数的前提下，如训练程序执行脚本run_train.sh,测试程序执行脚本run_eval.sh,环境变量env_set.sh等）

```
$ ./run_train.sh
```

run_train.sh内容如下：

```
source env_set.sh

nohup python -u train_image_classifier.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR \
  --checkpoint_path=$CHECKPOINT_PATH \
  --model_name=inception_v4 \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
  --train_dir=$TRAIN_DIR \
  --learning_rate=0.001 \
  --learning_rate_decay_factor=0.76\
  --num_epochs_per_decay=50 \
  --moving_average_decay=0.9999 \
  --optimizer=adam \
  --ignore_missing_vars=True \
  --batch_size=32 > output.log 2>&1 &
```

http://blog.csdn.net/lwplwf/article/details/76099010中讲了在后台执行程序，run_train.sh脚本文件中设置了后台执行，因此通过下面命令监控程序运行情况：

```
$ tail -f output.log # 当前日志动态显示
# 或者
$ cat output.log # 一次显示整个log文件
```

如下所示

```
INFO:tensorflow:Summary name /clone_loss is illegal; using clone_loss instead.
INFO:tensorflow:Fine-tuning from /home/lwp/pre_trained/inception_v4.ckpt
2017-07-27 08:32:08.547822: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547847: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547868: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547887: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.547892: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 08:32:08.861766: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-07-27 08:32:08.862322: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:01:00.0
Total memory: 10.91GiB
Free memory: 10.58GiB
2017-07-27 08:32:08.862342: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-07-27 08:32:08.862350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-07-27 08:32:08.862359: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
INFO:tensorflow:Restoring parameters from /home/lwp/pre_trained/inception_v4.ckpt
INFO:tensorflow:Starting Session.
INFO:tensorflow:Saving checkpoint to path /tmp/my_train_20170725/model.ckpt
INFO:tensorflow:Starting Queues.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 1.
INFO:tensorflow:global step 10: loss = 2.9544 (0.277 sec/step)
INFO:tensorflow:global step 20: loss = 2.7159 (0.267 sec/step)
INFO:tensorflow:global step 30: loss = 3.0572 (0.261 sec/step)
```

在/tmp/my_train_20170725路径下可以看到训练生成的checkpoint：meta、data、index

![这里写图片描述](http://img.blog.csdn.net/20170727084611516?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

该路径在环境变量设置脚本env_set.sh中定义

 **运行测试脚本** 

```
$ ./run_eval.sh
```

run_eval.sh的内容如下：

```

source env_set.sh
python -u eval_image_classifier.py \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR \
  --dataset_split_name=validation \
  --model_name=inception_v4 \
  --checkpoint_path=$TRAIN_DIR \
  --eval_dir=/tmp/eval/validation \
  --eval_interval_secs=60 \
  --batch_size=32
``` 

其中eval_interval_secs=60是指定两次验证的最小间隔时间为60s，具体定义在eval_image_classifier.py文件中。

这里训练和验证程序是分开的，模型在刚开始训练的时候效果必然很差，并不需要去验证，而且训练过程持续时间很长，如果将训练和验证放在一起的话，无用的验证就占用的很多时间。 
将训练和验证分开这样就可以在其他机器上访问checkpoint（路径为/tmp/my_train_20170725）去做验证，这样就可以把资源分散开。

执行后如下：

```
.
.
.
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:01:00.0
Total memory: 10.91GiB
Free memory: 2.24GiB
2017-07-27 09:27:33.151287: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-07-27 09:27:33.151292: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-07-27 09:27:33.151299: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
INFO:tensorflow:Restoring parameters from /tmp/my_train_20170725/model.ckpt-11028
INFO:tensorflow:Starting evaluation at 2017-07-27-01:27:47
2017-07-27 09:27:49.207742: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.51GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
INFO:tensorflow:Evaluation [1/12]
INFO:tensorflow:Evaluation [2/12]
INFO:tensorflow:Evaluation [3/12]
INFO:tensorflow:Evaluation [4/12]
INFO:tensorflow:Evaluation [5/12]
INFO:tensorflow:Evaluation [6/12]
INFO:tensorflow:Evaluation [7/12]
INFO:tensorflow:Evaluation [8/12]
INFO:tensorflow:Evaluation [9/12]
INFO:tensorflow:Evaluation [10/12]
INFO:tensorflow:Evaluation [11/12]
INFO:tensorflow:Evaluation [12/12]
INFO:tensorflow:Finished evaluation at 2017-07-27-01:27:56
2017-07-27 09:27:57.363998: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]
2017-07-27 09:27:57.364187: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.87760419]
INFO:tensorflow:Waiting for new checkpoint at /tmp/my_train_20170725
```

 **循环验证**  

可以看到给出了验证结果，注意最后一行Waiting for new checkpoint at /tmp/my_train_20170725，这是在eval_image_classifier.py中自定义了一个loop，去监听/tmp/my_train_20170725，一旦有新的checkpoint生成，就去执行一次验证。
 
**可视化训练：TensorBoard** 

执行：

```
$ tensorboard --logdir /tmp/my_train_20170725
```

得到：

```
Starting TensorBoard 55 at http://lw:6006
(Press CTRL+C to quit)
```

查看本机IP：

```
$ ifconfig -a
```

在浏览器中输入地址：

```
http://192.168.0.102：6006
```

![这里写图片描述](http://img.blog.csdn.net/20170727094702590?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

如果出现TensorBoard但不显示内容的情况，可以尝试换一个浏览器，我用Fire fox就是不显示，换chrome就好了。

 **结束训练** 

查看Python进程 
执行：

```
$ ps -ef |grep python
```

得到：

```
lwp       2780  2025 99 08:31 pts/0    03:38:22 python -u train_image_classifier.py --dataset_name=my_flower_5 --dataset_dir=/home/lwp/data/flower --checkpoint_path=/home/lwp/pre_trained/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/tmp/my_train_20170725 --learning_rate=0.001 --learning_rate_decay_factor=0.76 --num_epochs_per_decay=50 --moving_average_decay=0.9999 --optimizer=adam --ignore_missing_vars=True --batch_size=32
lwp      18830  3674  1 09:40 pts/2    00:00:15 /usr/bin/python /usr/local/bin/tensorboard --logdir /tmp/my_train_20170725
lwp      24837  2763  0 09:53 pts/0    00:00:00 grep --color=auto python
```

可以看到模型训练的进程号为2780

杀掉进程，结束训练

```
$ kill 2780
```

 **模型导出和使用** 

 **模型导出** 
 
执行脚本：

```
$ ./export_freeze.sh
```

得到3个文件： 
![这里写图片描述](http://img.blog.csdn.net/20170727100046250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题") 

分别存储的是模型的label、权重、结构

export_freeze.sh文件内容如下：

```
source env_set.sh
python -u export_inference_graph.py \
  --model_name=inception_v4 \
  --output_file=./my_inception_v4.pb \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR


NEWEST_CHECKPOINT=$(ls -t1 $TRAIN_DIR/model.ckpt*| head -n1)
NEWEST_CHECKPOINT=${NEWEST_CHECKPOINT%.*}
python -u ~/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=my_inception_v4.pb \
  --input_checkpoint=$NEWEST_CHECKPOINT \
  --output_graph=./my_inception_v4_freeze.pb \
  --input_binary=True \
  --output_node_name=InceptionV4/Logits/Predictions

cp $DATASET_DIR/labels.txt ./my_inception_v4_freeze.label
```
 
**模型使用**  
基于python的webserver 
执行脚本：

```
$ ./server.sh
```

得到：

```
listening on port 5001
2017-07-27 10:04:54.279779: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279800: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279806: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279810: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.279814: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-07-27 10:04:54.411389: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-07-27 10:04:54.411804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.582
pciBusID 0000:01:00.0
Total memory: 10.91GiB
Free memory: 10.50GiB
2017-07-27 10:04:54.411818: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-07-27 10:04:54.411822: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-07-27 10:04:54.411828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
 * Running on http://0.0.0.0:5001/ (Press CTRL+C to quit)
```

在浏览器输入地址：

```
http://本机IP:5001
```

![这里写图片描述](http://img.blog.csdn.net/20170727101054212?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

选择一张图片并上传，然后就会显示识别结果 
（注意，图片所在路径为/tmp/upload，在server.sh文件中定义）

server.sh文件内容如下：

```
python -u server.py \
  --model_name=my_inception_v4_freeze.pb \
  --label_file=my_inception_v4_freeze.label \
  --upload_folder=/tmp/uploadpython -u server.py \
  --model_name=my_inception_v4_freeze.pb \
  --label_file=my_inception_v4_freeze.label \
  --upload_folder=/tmp/upload
```

具体定义在server.py文件中

![这里写图片描述](http://img.blog.csdn.net/20170727101145271?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

如图得到5个分类的得分值，识别为sunflowers的score为0.79741

一些思考：我们刚才做的是5分类，分别是几种花，如果我们现在有一张猫的图片，这张图片对模型数据来说是未标识的，也就是对未标识的物体进行预测会是什么结果？ 
我们来试一下： 
![这里写图片描述](http://img.blog.csdn.net/20170727110209377?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "在这里输入图片标题")

可以看到，同样也给出了分类预测的得分值，可是这只猫当然不是蒲公英，这也是目前图像识别模型普遍存在的问题，也就是它不知道自己不知道。对人类而言，对于这5类花的预测分类，如果碰见这只猫，我们会说这不是花，或者遇见一种不认识的不属于这5类的我们会说我们不认识，或者不属于这5类，但是对于模型而言，它目前做不到，它最终只会把这只猫分到其中某一类花里面去。

----------------------------------------------------------------------------------------------------------------------------------------
# TensorFlow-Slim image classification library

[TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
is a new lightweight high-level API of TensorFlow (`tensorflow.contrib.slim`)
for defining, training and evaluating complex
models. This directory contains
code for training and evaluating several widely used Convolutional Neural
Network (CNN) image classification models using TF-slim.
It contains scripts that will allow
you to train models from scratch or fine-tune them from pre-trained network
weights. It also contains code for downloading standard image datasets,
converting them
to TensorFlow's native TFRecord format and reading them in using TF-Slim's
data reading and queueing utilities. You can easily train any model on any of
these datasets, as we demonstrate below. We've also included a
[jupyter notebook](https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb),
which provides working examples of how to use TF-Slim for image classification.

## Contacts

Maintainers of TF-slim:

* Nathan Silberman,
  github: [nathansilberman](https://github.com/nathansilberman)
* Sergio Guadarrama, github: [sguada](https://github.com/sguada)

## Table of contents

<a href="#Install">Installation and setup</a><br>
<a href='#Data'>Preparing the datasets</a><br>
<a href='#Pretrained'>Using pre-trained models</a><br>
<a href='#Training'>Training from scratch</a><br>
<a href='#Tuning'>Fine tuning to a new task</a><br>
<a href='#Eval'>Evaluating performance</a><br>
<a href='#Export'>Exporting Inference Graph</a><br>
<a href='#Troubleshooting'>Troubleshooting</a><br>

# Installation
<a id='Install'></a>

In this section, we describe the steps required to install the appropriate
prerequisite packages.

## Installing latest version of TF-slim

TF-Slim is available as `tf.contrib.slim` via TensorFlow 1.0. To test that your
installation is working, execute the following command; it should run without
raising any errors.

```
python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"
```

## Installing the TF-slim image models library

To use TF-Slim for image classification, you also have to install
the [TF-Slim image models library](https://github.com/tensorflow/models/tree/master/slim),
which is not part of the core TF library.
To do this, check out the
[tensorflow/models](https://github.com/tensorflow/models/) repository as follows:

```bash
cd $HOME/workspace
git clone https://github.com/tensorflow/models/
```

This will put the TF-Slim image models library in `$HOME/workspace/models/slim`.
(It will also create a directory called
[models/inception](https://github.com/tensorflow/models/tree/master/inception),
which contains an older version of slim; you can safely ignore this.)

To verify that this has worked, execute the following commands; it should run
without raising any errors.

```
cd $HOME/workspace/models/slim
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
```


# Preparing the datasets
<a id='Data'></a>

As part of this library, we've included scripts to download several popular
image datasets (listed below) and convert them to slim format.

Dataset | Training Set Size | Testing Set Size | Number of Classes | Comments
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
Flowers|2500 | 2500 | 5 | Various sizes (source: Flickr)
[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) | 60k| 10k | 10 |32x32 color
[MNIST](http://yann.lecun.com/exdb/mnist/)| 60k | 10k | 10 | 28x28 gray
[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 | Various sizes

## Downloading and converting to TFRecord format

For each dataset, we'll need to download the raw data and convert it to
TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format. Each TFRecord contains a
[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
protocol buffer. Below we demonstrate how to do this for the Flowers dataset.

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

You can use the same script to create the mnist and cifar10 datasets.
However, for ImageNet, you have to follow the instructions
[here](https://github.com/tensorflow/models/blob/master/inception/README.md#getting-started).
Note that you first have to sign up for an account at image-net.org.
Also, the download can take several hours, and could use up to 500GB.


## Creating a TF-Slim Dataset Descriptor.

Once the TFRecord files have been created, you can easily define a Slim
[Dataset](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/contrib/slim/python/slim/data/dataset.py),
which stores pointers to the data file, as well as various other pieces of
metadata, such as the class labels, the train/test split, and how to parse the
TFExample protos. We have included the TF-Slim Dataset descriptors
for
[Cifar10](https://github.com/tensorflow/models/blob/master/slim/datasets/cifar10.py),
[ImageNet](https://github.com/tensorflow/models/blob/master/slim/datasets/imagenet.py),
[Flowers](https://github.com/tensorflow/models/blob/master/slim/datasets/flowers.py),
and
[MNIST](https://github.com/tensorflow/models/blob/master/slim/datasets/mnist.py).
An example of how to load data using a TF-Slim dataset descriptor using a
TF-Slim
[DatasetDataProvider](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/dataset_data_provider.py)
is found below:

```python
import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
```


# Pre-trained Models
<a id='Pretrained'></a>

Neural nets work best when they have many parameters, making them powerful
function approximators.
However, this  means they must be trained on very large datasets. Because
training models from scratch can be a very computationally intensive process
requiring days or even weeks, we provide various pre-trained models,
as listed below. These CNNs have been trained on the
[ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/)
image classification dataset.

In the table below, we list each model, the corresponding
TensorFlow model file, the link to the model checkpoint, and the top 1 and top 5
accuracy (on the imagenet test set).
Note that the VGG and ResNet V1 parameters have been converted from their original
caffe formats
([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
and
[here](https://github.com/KaimingHe/deep-residual-networks)),
whereas the Inception and ResNet V2 parameters have been trained internally at
Google. Also be aware that these accuracies were computed by evaluating using a
single image crop. Some academic papers report higher accuracy by using multiple
crops at multiple scales.

Model | TF-Slim File | Checkpoint | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:--------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v1.py)|[inception_v1_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)|69.8|89.6|
[Inception V2](http://arxiv.org/abs/1502.03167)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v2.py)|[inception_v2_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz)|73.9|91.8|
[Inception V3](http://arxiv.org/abs/1512.00567)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v3.py)|[inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)|78.0|93.9|
[Inception V4](http://arxiv.org/abs/1602.07261)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v4.py)|[inception_v4_2016_09_09.tar.gz](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)|80.2|95.2|
[Inception-ResNet-v2](http://arxiv.org/abs/1602.07261)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)|[inception_resnet_v2_2016_08_30.tar.gz](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)|80.4|95.3|
[ResNet V1 50](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py)|[resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|75.2|92.2|
[ResNet V1 101](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py)|[resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)|76.4|92.9|
[ResNet V1 152](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py)|[resnet_v1_152_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)|76.8|93.2|
[ResNet V2 50](https://arxiv.org/abs/1603.05027)^|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py)|[resnet_v2_50_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)|75.6|92.8|
[ResNet V2 101](https://arxiv.org/abs/1603.05027)^|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py)|[resnet_v2_101_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)|77.0|93.7|
[ResNet V2 152](https://arxiv.org/abs/1603.05027)^|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py)|[resnet_v2_152_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)|77.8|94.1|
[ResNet V2 200](https://arxiv.org/abs/1603.05027)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py)|[TBA]()|79.9\*|95.2\*|
[VGG 16](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/vgg.py)|[vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)|71.5|89.8|
[VGG 19](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/vgg.py)|[vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)|71.1|89.8|
[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.py)|[mobilenet_v1_1.0_224_2017_06_14.tar.gz](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz)|70.7|89.5|
[MobileNet_v1_0.50_160](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.50_160_2017_06_14.tar.gz](http://download.tensorflow.org/models/mobilenet_v1_0.50_160_2017_06_14.tar.gz)|59.9|82.5|
[MobileNet_v1_0.25_128](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.25_128_2017_06_14.tar.gz](http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz)|41.3|66.2|

^ ResNet V2 models use Inception pre-processing and input image size of 299 (use
`--preprocessing_name inception --eval_image_size 299` when using
`eval_image_classifier.py`). Performance numbers for ResNet V2 models are
reported on ImageNet valdiation set.

All 16 MobileNet Models reported in the [MobileNet Paper](https://arxiv.org/abs/1704.04861) can be found [here](https://github.com/tensorflow/models/tree/master/slim/nets/mobilenet_v1.md).

(\*): Results quoted from the [paper](https://arxiv.org/abs/1603.05027).

Here is an example of how to download the Inception V3 checkpoint:

```shell
$ CHECKPOINT_DIR=/tmp/checkpoints
$ mkdir ${CHECKPOINT_DIR}
$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
$ tar -xvf inception_v3_2016_08_28.tar.gz
$ mv inception_v3.ckpt ${CHECKPOINT_DIR}
$ rm inception_v3_2016_08_28.tar.gz
```



# Training a model from scratch.
<a id='Training'></a>

We provide an easy way to train a model from scratch using any TF-Slim dataset.
The following example demonstrates how to train Inception V3 using the default
parameters on the ImageNet dataset.

```shell
DATASET_DIR=/tmp/imagenet
TRAIN_DIR=/tmp/train_logs
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3
```

This process may take several days, depending on your hardware setup.
For convenience, we provide a way to train a model on multiple GPUs,
and/or multiple CPUs, either synchrononously or asynchronously.
See [model_deploy](https://github.com/tensorflow/models/blob/master/slim/deployment/model_deploy.py)
for details.


# Fine-tuning a model from an existing checkpoint
<a id='Tuning'></a>

Rather than training from scratch, we'll often want to start from a pre-trained
model and fine-tune it.
To indicate a checkpoint from which to fine-tune, we'll call training with
the `--checkpoint_path` flag and assign it an absolute path to a checkpoint
file.

When fine-tuning a model, we need to be careful about restoring checkpoint
weights. In particular, when we fine-tune a model on a new task with a different
number of output labels, we wont be able restore the final logits (classifier)
layer. For this, we'll use the `--checkpoint_exclude_scopes` flag. This flag
hinders certain variables from being loaded. When fine-tuning on a
classification task using a different number of classes than the trained model,
the new model will have a final 'logits' layer whose dimensions differ from the
pre-trained model. For example, if fine-tuning an ImageNet-trained model on
Flowers, the pre-trained logits layer will have dimensions `[2048 x 1001]` but
our new logits layer will have dimensions `[2048 x 5]`. Consequently, this
flag indicates to TF-Slim to avoid loading these weights from the checkpoint.

Keep in mind that warm-starting from a checkpoint affects the model's weights
only during the initialization of the model. Once a model has started training,
a new checkpoint will be created in `${TRAIN_DIR}`. If the fine-tuning
training is stopped and restarted, this new checkpoint will be the one from
which weights are restored and not the `${checkpoint_path}$`. Consequently,
the flags `--checkpoint_path` and `--checkpoint_exclude_scopes` are only used
during the `0-`th global step (model initialization). Typically for fine-tuning
one only want train a sub-set of layers, so the flag `--trainable_scopes` allows
to specify which subsets of layers should trained, the rest would remain frozen.

Below we give an example of
[fine-tuning inception-v3 on flowers](https://github.com/tensorflow/models/blob/master/slim/scripts/finetune_inception_v3_on_flowers.sh),
inception_v3  was trained on ImageNet with 1000 class labels, but the flowers
dataset only have 5 classes. Since the dataset is quite small we will only train
the new layers.


```shell
$ DATASET_DIR=/tmp/flowers
$ TRAIN_DIR=/tmp/flowers-models/inception_v3
$ CHECKPOINT_PATH=/tmp/my_checkpoints/inception_v3.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
```



# Evaluating performance of a model
<a id='Eval'></a>

To evaluate the performance of a model (whether pretrained or your own),
you can use the eval_image_classifier.py script, as shown below.

Below we give an example of downloading the pretrained inception model and
evaluating it on the imagenet dataset.

```shell
CHECKPOINT_FILE = ${CHECKPOINT_DIR}/inception_v3.ckpt  # Example
$ python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3
```


# Exporting the Inference Graph
<a id='Export'></a>

Saves out a GraphDef containing the architecture of the model.

To use it with a model name defined by slim, run:

```shell
$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inf_graph.pb

$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/tmp/mobilenet_v1_224.pb
```

## Freezing the exported Graph
If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

```shell
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

```shell
bazel build tensorflow/tools/graph_transforms:summarize_graph

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=/tmp/inception_v3_inf_graph.pb
```

## Run label image in C++

To run the resulting graph in C++, you can look at the label_image sample code:

```shell
bazel build tensorflow/examples/label_image:label_image

bazel-bin/tensorflow/examples/label_image/label_image \
  --image=${HOME}/Pictures/flowers.jpg \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph=/tmp/frozen_inception_v3.pb \
  --labels=/tmp/imagenet_slim_labels.txt \
  --input_mean=0 \
  --input_std=255 \
  --logtostderr
```


# Troubleshooting
<a id='Troubleshooting'></a>

#### The model runs out of CPU memory.

See
[Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/master/inception#the-model-runs-out-of-cpu-memory).

#### The model runs out of GPU memory.

See
[Adjusting Memory Demands](https://github.com/tensorflow/models/tree/master/inception#adjusting-memory-demands).

#### The model training results in NaN's.

See
[Model Resulting in NaNs](https://github.com/tensorflow/models/tree/master/inception#the-model-training-results-in-nans).

#### The ResNet and VGG Models have 1000 classes but the ImageNet dataset has 1001

The ImageNet dataset provided has an empty background class which can be used
to fine-tune the model to other tasks. If you try training or fine-tuning the
VGG or ResNet models using the ImageNet dataset, you might encounter the
following error:

```bash
InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [1001] rhs shape= [1000]
```
This is due to the fact that the VGG and ResNet V1 final layers have only 1000
outputs rather than 1001.

To fix this issue, you can set the `--labels_offset=1` flag. This results in
the ImageNet labels being shifted down by one:


#### I wish to train a model with a different image size.

The preprocessing functions all take `height` and `width` as parameters. You
can change the default values using the following snippet:

```python
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    height=MY_NEW_HEIGHT,
    width=MY_NEW_WIDTH,
    is_training=True)
```

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).

