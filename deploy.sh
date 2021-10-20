# !/bin/bash

# argument list: remote ip, model name, n_classes, dataset name, dataset path, n_batch, version_number, 
remote_username="hhameed"
remote_ip="10.249.3.13"

model_folder="/home/pbr-student/pytorch-YOLOv4/"
model_path="yolov4.pth"
sample_image="/home/pbr-student/val2017/000000000139.jpg"
batch_size=4
n_classes=80
img_height=512
img_length=512

model_name="YOLOv4"
model_version=4

remote_model_repo="~/model_repository/"

# prepare config file
config="
name: \"YOLOv4\"
platform: \"onnxruntime_onnx\"
max_batch_size : 0
input [
  {
    name: \"input\"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [$batch_size, 512, 512]
    reshape { shape: [$batch_size, 3, 512, 512] }
  }
]
output [
  {
    name: \"confs\"
    data_type: TYPE_FP32
    dims: [$batch_size, 16128, 80]
  },
  {
    name: \"boxes\"
    data_type: TYPE_FP32
    dims: [$batch_size, 16128, 1, 4]
  }
]
cc_model_filenames: {}
metric_tags: {}
parameters: {}
model_warmup: []
"

# convert the model
cd $model_folder
python3 demo_pytorch2onnx.py $model_path /home/pbr-student/val2017/000000000139.jpg $batch_size $n_classes $img_height $img_length

# prepare directory
ssh $remote_username@$remote_ip "mkdir $remote_model_repo$model_name/$model_version/"

# copy model
onnx_file=$(ls ../pytorch-YOLOv4/ -tl | head -3 | tail -1 | awk '{print $9}')
scp "$model_folder$onnx_file" "${remote_username}@${remote_ip}:${remote_model_repo}${model_name}/${model_version}"

# create the config file
ssh $remote_username@$remote_ip "> config.pbtxt;echo -e \"$config\" >> $remote_model_repo$model_name/config.pbtxt"

