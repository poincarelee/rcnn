import os
from preprocess.imdb import imdb
import numpy as np
import xml.dom.minidom as minidom
import pickle
import utils.utils as utils
import skimage.io
import selective_search.selective_search as ss
import scipy.sparse
import config

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
        self.cache_path = config._cache_path
        # image set file names
        self.image_index = self.load_image_set_index()
        # roidb definition
        # key and value(may not include all the keys)
        # boxes: box location, (box_num, 4)
        # gt_overlaps: all boxes' scores in different classes, (box_num, class_num)
        # gt_classes: all boxes' true label, (box_num,)
        # flipped: whether flipped
        # image: image path
        # width: image width
        # height: image height
        # max_overlaps: each box's maximum score in all classes, (box_num,)
        # max_classes: each box's label which has maximum score, (box_num,)
        # bbox_targets: each box's label and the nearest ground truth's box location, (box_num, 5) => (c, tx, ty, tw, th)
        # gt_idx: each box's nearest ground truth's index, (box_num,)
        self.rois = self.load_rois()

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
        # image path
        img_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')

        # parse xml file
        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        # x, y, w, h, label
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        gt_overlaps = np.zeros((num_objs, len(self.classes)), dtype=np.float32)
        max_classes = np.zeros((num_objs), dtype=np.int32)
        # with itself overlap is 1.0
        max_overlaps = np.ones((num_objs), dtype=np.float32)
        gt_idxs = np.zeros((num_objs), dtype=np.int32)

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
            gt_overlaps[i, cls] = 1.0
            max_classes[i] = cls
            gt_idxs[i] = i
            # load the original image
            # img = skimage.io.imread(img_path)
            # cliped_img = utils.clip_pic(img, boxes[i, :])
            # resized_img = utils.resize_img(cliped_img, 227, 227)
            # float_img = np.asarray(resized_img, dtype=np.float32)

        gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)
        return {
            'image': img_path,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': gt_overlaps,
            'max_classes': max_classes,
            'max_overlaps': max_overlaps,
            'gt_idxs': gt_idxs,
        }

    def load_ss_rois(self):
        # load the selective search regions of image set
        cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
        if os.path.exists(cache_file):
            print('Load ss roidb from {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                ss_roidb = pickle.load(f)
            print('Load finished')
            return ss_roidb
        ss_roidb = []
        for i in range(len(self.image_index)):
            ss_roi = self.load_ss_roi(self.image_index[i])
            ss_roidb.append(ss_roi)
            utils.view_bar('Load ss roidb from {}'.format(self.image_index[i] + '.xml'), i, len(self.image_index))
        with open(cache_file, 'wb') as f:
            pickle.dump(ss_roidb, f, pickle.HIGHEST_PROTOCOL)
        print('Write ss roidb to {}'.format(cache_file))
        return ss_roidb

    def load_ss_roi(self, index):
        # load the selective search regions of one image
        img_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        boxes = []
        img = skimage.io.imread(img_path)
        _, regions = ss.selective_search(img, scale=500, sigma=0.9, min_size=10)
        gt_overlaps = np.zeros((len(regions), len(self.classes)), dtype=np.float32)
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
            # cliped_img = utils.clip_pic(img, r['rect'])
            # resized_img = utils.resize_img(img, 227, 227)
            # float_img = np.asarray(resized_img, dtype=np.float32)

            for j, t in enumerate(zip(gt_roi['boxes'], gt_roi['gt_classes'])):
                # t[0] box, t[1] gt_class
                iou = utils.IOU(r['rect'], t[0])
                iou_boxes[j][i] = iou
        # get the maximum iou index for each column
        argmaxes = iou_boxes.argmax(axis=0)
        # get the maximum iou for each column
        maxes = iou_boxes.max(axis=0)
        I = np.where(maxes > 0)[0]
        gt_overlaps[I, gt_roi['gt_classes'][argmaxes[I]]] = maxes[I]
        max_classes = gt_overlaps.argmax(axis=1)
        max_overlaps = gt_overlaps.max(axis=1)
        gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)
        return {
            'image': img_path,
            'boxes': np.array(boxes, dtype=np.float32),
            # to discriminate this is selective search region
            'gt_classes': np.zeros((len(regions)), dtype=np.int32),
            'gt_overlaps': gt_overlaps,
            'max_classes': max_classes,
            'max_overlaps': max_overlaps,
            'gt_idxs': argmaxes,
        }

    def load_rois(self):
        # load both the ground truth and the selective search regions of image set
        gt_rois = self.load_gt_rois()
        ss_rois = self.load_ss_rois()
        # merge them
        # ground truth are followed by selective search regions
        for i in range(len(gt_rois)):
            # merge boxes by row
            gt_rois[i]['boxes'] = np.vstack((gt_rois[i]['boxes'], ss_rois[i]['boxes']))
            # merge gt_classes by column
            gt_rois[i]['gt_classes'] = np.hstack((gt_rois[i]['gt_classes'], ss_rois[i]['gt_classes']))
            # merge gt_overlaps by row
            gt_rois[i]['gt_overlaps'] = scipy.sparse.vstack([gt_rois[i]['gt_overlaps'], ss_rois[i]['gt_overlaps']])
            # merge max_classes by column
            gt_rois[i]['max_classes'] = np.hstack((gt_rois[i]['max_classes'], ss_rois[i]['max_classes']))
            # merge max_overlaps by column
            gt_rois[i]['max_overlaps'] = np.hstack((gt_rois[i]['max_overlaps'], ss_rois[i]['max_overlaps']))
            # merge gt_idxs by column
            gt_rois[i]['gt_idxs'] = np.hstack((gt_rois[i]['gt_idxs'], ss_rois[i]['gt_idxs']))
        return gt_rois







