import sys
import os
import config
sys.path.insert(0, os.path.join(config._caffe_root, 'python'))

from preprocess.pascal_voc import pascal_voc
import utils.utils as utils
from train import SolverWrapper

pv = pascal_voc('train', '2007', './VOCdevkit')

print(pv.image_index)

gt_roi = pv.load_gt_roi('000001')
labels = []
for i in gt_roi['gt_classes']:
    labels.append(pv.classes[i])

# create image set for train in lmdb format
utils.create_imdb('./lmdb/train_lmdb', pv.rois, 0.5)
# initialize SolverWrapper
sw = SolverWrapper('./models/solver.prototxt', pv.rois, './pretrained_models', pretrained_model=None)
sw.train_model(1)
utils.show_rect('./VOCdevkit/VOC2007/JPEGImages/000001.jpg', gt_roi['boxes'], labels)
