import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.transform
import skimage.io
import numpy as np
from caffe.proto import caffe_pb2
import lmdb
import config
import cv2

# show process bar
def view_bar(message, num, total):
    rate = float(num) / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, '>' * rate_num, ' ' * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

# show image with rect
def show_rect(path, regions, labels):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = plt.imread(path)
    ax.imshow(img)
    for region, label in zip(regions, labels):
        w = region[2] - region[0]
        h = region[3] - region[1]
        x = region[0]
        y = region[1]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        # add label text
        ax.text(x, y, label, fontdict={'size': 12, 'color': 'r'})
    plt.show()

# clip the image
def clip_pic(img, box):
    return img[box[1]:box[3], box[0]:box[2], :]

# resize the image
def resize_img(in_image, height, width, out_image=None):
    img = skimage.transform.resize(in_image, (height, width))

# if two boxes have intersection
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    flag = False
    # calculate the center coordinates
    xc_a = (xmin_a + xmax_a) / 2
    yc_a = (ymin_a + ymax_a) / 2
    xc_b = (xmin_b + xmax_b) / 2
    yc_b = (ymin_b + ymax_b) / 2
    if abs(xc_b - xc_a) < (xmax_a - xmin_a + xmax_b - xmin_a) / 2 and abs(yc_b - yc_a) < (
                ymax_a - ymin_a + ymax_b - ymin_b) / 2:
        flag = True
    if not flag:
        return flag
    # calculate the area intersected
    x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
    y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
    x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
    x_intersect_h = y_sorted_list[2] - y_sorted_list[1]
    area_inter = x_intersect_w * x_intersect_h
    return area_inter

# IOU
def IOU(box1, box2):
    area_inter = if_intersection(box1[0], box1[2], box1[1], box1[3],
                                 box2[0], box2[2], box2[1], box2[3])
    if area_inter:
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # calculate IOU
        iou = float(area_inter) / (area1 + area2 - area_inter)
        return iou
    return False

# add bbox_targets of roidb
def add_bbox_targets_db(roidb, threshold):
    class_counts = np.zeros((config._num_classes, 1), dtype=np.int32)
    sums = np.zeros((config._num_classes, 4), dtype=np.float32)
    squared_sums = np.zeros((config._num_classes, 4), dtype=np.float32)
    for rois in roidb:
        rois['bbox_targets'] = add_bbox_targets(rois, threshold)
        for cls in range(1, config._num_classes):
            cls_idxs = np.where(rois['bbox_targets'][:, 0] == cls)[0]
            class_counts[cls] += cls_idxs.size
            sums[cls, :] += rois['bbox_targets'][cls_idxs, 1:].sum(axis=0)
            squared_sums[cls, :] += (rois['bbox_targets'][cls_idxs, 1:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # normalize targets
    for rois in roidb:
        targets = rois['bbox_targets']
        for cls in range(1, config._num_classes):
            cls_idxs = np.where(targets[:, 0] == cls)[0]
            rois['bbox_targets'][cls_idxs, 1:] -= means[cls, :]
            rois['bbox_targets'][cls_idxs, 1:] /= stds[cls, :]

    return means.ravel(), stds.ravel()

# add bbox_targets of rois(one image)
def add_bbox_targets(rois, threshold):
    # bbox regression training examples
    ex_idxs = np.where(rois['max_overlaps'] >= threshold)[0]
    # the nearest ground truth index for examples
    gt_idxs = rois['gt_idxs'][ex_idxs]
    gt_boxes = rois['boxes'][gt_idxs]
    ex_boxes = rois['boxes'][ex_idxs]

    ex_widths = ex_boxes[:, 2] - ex_boxes[:, 0]
    ex_heights = ex_boxes[:, 3] - ex_boxes[:, 1]
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((len(rois['gt_classes']), 5), dtype=np.float32)
    targets[ex_idxs, :] = [rois['max_classes'][ex_idxs], targets_dx, targets_dy, targets_dw, targets_dh]

    return targets

# # make datum, the unit of lmdb
# def make_datum(img, label):
#     # image is numpy.ndarray format. BGR instead of RGB
#     return caffe_pb2.Datum(
#         channels=3,
#         width=227,
#         height=227,
#         label=label,
#         # translate from HWC to CHW
#         data=np.rollaxis(img, 2).tobytes()
#     )
#
# # create imdb
# def create_imdb(lmdb_path, roidb, threshold):
#     in_db = lmdb.open(lmdb_path)
#     # create db handler
#     with in_db.begin(write=True) as in_txn:
#         count = 0
#         for i, rois in enumerate(roidb):
#             # read image
#             # cv2 read BGR
#             # so we needn't to translate RBG to BGR
#             img = cv2.imread(rois['image'])
#             max_classes = rois['max_classes']
#             max_overlaps = rois['max_overlaps']
#             idx = np.where(max_overlaps >= threshold)
#             labels = np.zeros((len(max_classes)), dtype=np.int32)
#             labels[idx] = max_classes[idx]
#             for j, box in enumerate(rois['boxes']):
#                 cliped_img = clip_pic(img, box)
#                 resized_img = resize_img(cliped_img, 227, 227)
#                 label = labels[j]
#
#                 datum = make_datum(resized_img, label)
#                 in_txn.put('{:0>5d}'.format(count), datum.SerializeToString())
#                 print('{:0>5d}'.format(count) + ':' + rois['image'])
#                 count += 1
#         in_db.close()





