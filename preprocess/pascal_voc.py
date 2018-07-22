import os
from preprocess.imdb import imdb
import numpy as np
import xml.dom.minidom as minidom
import pickle
import utils.utils as utils
import skimage.io
import selective_search.selective_search as ss
import scipy.sparse

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self.year = year
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = os.path.join(self.devkit_path, 'VOC' + self.year)
        self.classes = ('background', # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.classes_idx = dict(zip(self.classes, range(len(self.classes))))
        self.image_ext = '.jpg'
        # image set file names
        self.image_index = self.load_image_set_index()
        self.cache_path = '../cache'

    def load_image_set_index(self):
        # for example: VOCdevkit/VOC2007/ImageSets/Main/train.txt
        image_set_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        if not os.path.exists(image_set_file):
            print('Path does not exist: {}'.format(image_set_file))
        else:
            with open(image_set_file) as f:
                image_index = [x.strip() for x in f.readlines()]
            return image_index

    def load_gt_rois(self):
        # load the ground truth regions of image set
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            print('Load gt roidb from {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                gt_roidb = pickle.load(f)
            print('Load finished')
            return gt_roidb
        gt_roidb = []
        for i in range(len(self.image_index)):
            gt_roi = self.load_gt_roi(self.image_index[i])
            gt_roidb.append(gt_roi)
            utils.view_bar('Load gt roidb from {}'.format(self.image_index[i] + '.xml'), i, len(self.image_index))
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_roidb, f, pickle.HIGHEST_PROTOCOL)
        print('Write gt roidb to {}'.format(cache_file))
        return gt_roidb

    def load_gt_roi(self, index):
        # load the ground truth regions of one image
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')

        # parse xml file
        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        images = []
        # x, y, w, h, label
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, len(self.classes)), dtype=np.float32)

        for i, obj in enumerate(objs):
            x1 = float(obj.getElementsByTagName('xmin')[0].childNodes[0].data) - 1
            y1 = float(obj.getElementsByTagName('ymin')[0].childNodes[0].data) - 1
            x2 = float(obj.getElementsByTagName('xmax')[0].childNodes[0].data) - 1
            y2 = float(obj.getElementsByTagName('ymax')[0].childNodes[0].data) - 1
            # class name
            class_name = str(obj.getElementsByTagName('name')[0].childNodes[0].data).lower().strip()
            cls = self.classes_idx[class_name]
            boxes[i, :] = [x1, y1, x2, y2]
            gt_classes[i] = cls
            overlaps[i, cls] = 1.0
            # load the original image
            img_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            img = skimage.io.imread(img_path)
            cliped_img = utils.clip_pic(img, boxes[i, :])
            resized_img = utils.resize_img(cliped_img, 227, 227)
            float_img = np.asarray(resized_img, dtype=np.float32)
            images.append(float_img)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {
            'images': images,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
        }

    def load_ss_roi(self, index):
        # load the selective search regions of one image
        filename = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        images = []
        boxes = []
        img = skimage.io.imread(filename)
        _, regions = ss.selective_search(img, scale=500, sigma=0.9, min_size=10)
        overlaps = np.zeros((len(regions), len(self.classes)), dtype=np.float32)
        # ground truth
        gt_roi = self.load_gt_roi(index)
        # calculate iou between every region and every ground truth
        iou_boxes = np.zeros((len(gt_roi['boxes']), len(regions)), dtype=np.float32)

        candidates = set()
        for i, r in enumerate(regions):
            # remove the same box
            if r['rect'] in candidates:
                continue
            # remove the small box
            if r['size'] < 220:
                continue
            candidates.add(r['rect'])
            boxes.append(r['rect'])
            # resize the box
            cliped_img = utils.clip_pic(img, r['rect'])
            resized_img = utils.resize_img(img, 227, 227)
            float_img = np.asarray(resized_img, dtype=np.float32)
            images.append(float_img)

            for j, t in enumerate(zip(gt_roi['boxes'], gt_roi['gt_classes'])):
                # t[0] box, t[1] gt_class
                iou = utils.IOU(r['rect'], t[0])
                iou_boxes[j][i] = iou
        # get the maximum iou index for each column
        argmaxes = iou_boxes.argmax(axis=1)
        # get the maximum iou for each column
        maxes = iou_boxes.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_roi['gt_classes'][argmaxes[I]]] = maxes[I]

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {
            'images': images,
            'boxes': boxes,
            'gt_classes': np.zeros((len(regions),), dtype=np.int32),
            'gt_overlaps': overlaps,
        }








