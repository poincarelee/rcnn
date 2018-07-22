import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.transform

# show process bar
def view_bar(message, num, total):
    rate = num / total
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
                                 box2[0], box2[2], box2[1], bpx2[3])
    if area_inter:
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # calculate IOU
        iou = float(area_inter) / (area1 + area2 - area_inter)
        return iou
    return False