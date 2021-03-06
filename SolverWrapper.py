# -*- coding: utf-8 -*-
# @Time    : 7/23/2018 2:05 PM
# @Author  : Ruichen Shao
# @FileName: SolverWrapper.py.py
import caffe
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import config
import utils.utils as utils

class SolverWrapper(object):
    def __init__(self, solver_prototxt, roidb, output_dir, pretrained_model=None):
        self.output_dir = output_dir
        # load solver prototxt
        self.solver = caffe.SGDSolver(solver_prototxt)
        if not pretrained_model:
            print('Load pretrained model weights from {}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        print('Compute bounding box regression targets...')
        self.means, self.stds = utils.add_bbox_targets_db(roidb, config._finetune_threshold)
        # load solver parameters from file
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    # snapshot the model during training
    def snap_shot(self):
        net = self.solver.net
        # because bounding box is normalized to train
        # bounding box cannot be normalized when testing
        # we need to scale and shift the weights and biases

        # save original values to recover
        weights = net.params['bbox_pred'][0].data.copy()
        biases = net.params['bbox_pred'][1].data.copy()

        # do the scale and shift operation
        # I'm not sure whether it is correct

        net.params['bbox_pred'][1].data[...] = net.params['bbox_pred'][1].data * self.stds + self.means * net.params['bbox_pred'][0].data
        net.params['bbox_pred'][0].data[...] = net.params['bbox_pred'][0].data * self.stds

        if not os.path.exits(self.output_dir):
            os.makedirs(self.output_dir)
        filename = self.solver_param.snapshot_prefix + '_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        # recover the weights and biases
        net.params['bbox_pred'][0].data[...] = weights
        net.params['bbox_pred'][1].data[...] = biases

    # train the caffe model
    def train_model(self, max_iters):
        last_snapshot_iter = -1
        while self.solver.iter < max_iters:
            # make one SGD update
            self.solver.step(1)
            if self.solver.iter % config._snapshot_interval == 0:
                last_snapshot_iter = self.solver.iter
                self.snap_shot()
        if last_snapshot_iter != self.solver.iter:
            self.snap_shot()









