# -*- coding: utf-8 -*-
# @Time    : 7/23/2018 4:52 PM
# @Author  : Ruichen Shao
# @FileName: config.py.py
import numpy as np

_caffe_root = 'C:/caffe'
_cache_path = 'C:/Users/ruichenshao/PycharmProjects/rcnn/cache'
_snapshot_interval = 1000
_finetune_threshold = 0.5
_svm_threshold = 0.3
_num_classes = 21
_train_scales = (600,)
_train_max_size = 1000
_train_imgs_per_batch = 2

# Fraction of minibatch that is labeled foreground(class > 0)
_train_fg_fraction = 0.25
_train_fg_threshold = 0.5
# Background's iou is in [low, high)
_train_bg_threshold_hi = 0.5
_train_bg_threshold_lo = 0.1
_train_batch_size = 128

_pixel_means = np.array([[[102.9801, 115,9465, 122.7717]]])
_use_prefetch = True
_rng_seed = 3