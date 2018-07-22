# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# import sys
# caffe_root = '/home/hanfu/caffe'
# sys.path.insert(0, caffe_root + 'python')
#
# import caffe
# import os
# if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print 'CaffeNet found.'
# else:
#     print 'Downloading pre-trained CaffeNet model...'
#
# # pick the first GPU
# caffe.set_device(0)
# caffe.set_mode_gpu()
#
# model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
# model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#
# net = caffe.Net(model_def, model_weights, caffe.TEST)
#
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)
# print 'mean-subtracted values:', zip('BGR', mu)
#
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# # transform from (height, width, channel) to (channel, height, width)
# transformer.set_transpose('data', (2, 0, 1))
# # subtract the dataset-mean value in each channel
# transformer.set_mean('data', mu)
# # rescale from [0, 1] to [0, 255]
# transformer.set_raw_scale('data', 255)
# # swap channels from RGB to BGR
# transformer.set_channel_swap('data', (2, 1, 0))
#
# # (batch_size, channel_BGR, height, width)
# net.blobs['data'].reshape(50, 3, 227, 227)
# image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# # preprocess image into the form net accepted
# transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
#
# # copy the image data into the memory allocated for the net
# net.blobs['data'].data[...] = transformed_image
# # perform classification
# output = net.forward()
# # the output probability vector for the first image in the batch
# output_prob = output['prob'][0]
# print 'predicted class is:', output_prob.argmax()
#
# # load ImageNet labels
# labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
# if not os.path.exists(labels_file):
#     print 'Please run data/ilsvrc12/get_ilsvrc_aux.sh'
#
# labels = np.loadtxt(labels_file, str, delimiter='\t')
# print 'output label:', labels[output_prob.argmax]
#
# # sort top five predictions from softmax output
# # reverse and then select top five
# top_idxs = output_prob.argsort()[::-1][:5]
# print 'probabilities and labels:'
# zip(output_prob[top_idxs], labels[top_idxs])
#
# # for each layer, show the output shape
# for layer_name, blob in net.blobs.iteritems():
#     print layer_name + '\t' + str(blob.data.shape)
# for layer_name, param in net.params.iteritems():
#     print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
#
# def vis_square(data, save_path):
#     # data shape (n, height, width) or (n, height, width, 3)
#     # normalize data for display
#     data = (data - data.min()) / (data.max() - data.min())
#     # force the number of filters to be square
#     n = int(np.ceil(np.sqrt(data.shape[0])))
#     padding = (((0, n ** 2 - data.shape[0]),
#                 (0, 1), (0, 1))                 # add some space between filters
#                + ((0, 0),) * (data.ndim - 3))   # don't pad the channel dimension
#     data = np.pad(data, padding, mode='constant', constant_values=1)
#
#     # title the filters into an image
#     data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#     data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#
#     plt.imshow(data)
#     plt.axis('off')
#     plt.savefig(save_path)
#
# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1), 'filters.png')
# feat = net.blobs['conv1'].data[0, :36]
# vis_square(feat, 'features.png')

from preprocess.pascal_voc import pascal_voc
import utils.utils as utils

pascal = pascal_voc('train', '2007', './VOCdevkit')

gt_roi = pascal.load_gt_roi('000001')
labels = []
for i in gt_roi['gt_classes']:
    labels.append(pascal.classes[i])

utils.show_rect('./VOCdevkit/VOC2007/JPEGImages/000001.jpg', gt_roi['boxes'], labels)
