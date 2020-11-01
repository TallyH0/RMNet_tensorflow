#Image Info
img_h = 128
img_w = 64
img_c = 3

#Training Info
batch_size = 64
max_epoch = 2500

#Loss & Model Info
activation = 'leaky'
weight_decay = 0
factor_center_loss = 1
alpha_center_loss = 0.95
factor_push_loss = 1
factor_gpush_loss = 1

#Dataset Info
txt_dataset = 'market1501_train.txt'
model_dir = 'model_ped_reid'

#Random erasing parameter
param_erasing = {
    'prob' : 0.5,
    'sl' : 0.02,
    'sh' : 0.2,
    'r1' : 0.3,
    'r2' : 1/0.3
}
