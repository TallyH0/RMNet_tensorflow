# RMNet_tensorflow
This is a tensorflow implementation of [Fast and Accurate Person Re-Identificationwith RMNet](https://arxiv.org/pdf/1812.02465.pdf)

# Requirement
* Tensorflow 1.13
* Opencv-python

# Dataset
you can download market1501 dataset [here](https://drive.google.com/file/d/1wb4UHGDSvI4kkGjsfuqYXiTerBttEYh-/view?usp=sharing)

# Result
you can download pre-trained model [here](https://drive.google.com/file/d/1vNRKIUmuOXqWaxSY2HjixmgYvTfifkxM/view?usp=sharing)
rank@1 accuracy : 91.02%

# Train
    python train.py --config configuration.py --data_dir <market1501 dataset dir>

# Test
    python test.py --config cfg_pretrained.py --data_dir <market1501 dataset dir> --txt_query market1501_query.txt --txt_test market1501_test.txt

# TODO
* add mAP accuracy
