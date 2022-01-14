Requirements:  
ubuntu 16.04  
python 3.5  
tensorflow 1.13.2  
snpe-1.43  
  
  
Guide:  
git clone https://github.com/tensorflow/models.git -b r1.13.0  
cd models/research  
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip  
unzip protobuf.zip  
./bin/protoc object_detection/protos/*.proto --python_out=.  
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim  
python object_detection/builders/model_builder_test.py  

copy all the scripts in home directory  
run the make scripts in home directory  
