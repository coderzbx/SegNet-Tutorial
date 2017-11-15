# -*-coding:utf-8-*-

import numpy as np
import argparse
import os
from os import listdir
import sys
import collections
import cv2

from Mapillary.mapillary_label import mapillary_instance_labels

# label_dir：要计算class_weighting的图片，一般是instances，或者annotations
# image_dir：要打分类标记的图片
# class_dir：要保存分类标记的图片的目录
# limit：一个分类最多保存的图片

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--label_dir', type=str, help='Path to the folder containing the images with annotations')
parser.add_argument('--image_dir', type=str)
parser.add_argument('--class_dir', type=str)
parser.add_argument('--limit', type=int)
args = parser.parse_args()

symbol_size = 50

label_dir = args.label_dir
class_dir = args.class_dir
limit = args.limit
if limit <= 0:
    limit = 5
image_dir = args.image_dir

images = listdir(label_dir)
# images = [images[i] for i in range(0, limit)]
# Keep only images and append image_names to directory
image_list = [os.path.join(label_dir, s) for s in images if s.lower().endswith(('.png', '.jpg', '.jpeg'))]
print("Number of images:%d" % len(image_list))

# id->class_name
id_name_dict = {l.id: l.name for l in mapillary_instance_labels}

# id=>[{filename=>pixel_coord}],limit
id_labeled_dict = {l.id: [] for l in mapillary_instance_labels}

# file_name=>total_pixel_count
file_pixel_dict = {}

# id=>{file_name=>pixel_count}
id_pixel_dict = {l.id: {} for l in mapillary_instance_labels}

# id=>file_count
id_file_count = {l.id: 0 for l in mapillary_instance_labels}


def get_class_per_image(img):
    name_list = str(img).split("/")
    file_id_ = name_list[len(name_list) - 1]

    dic_class_pixelcount = dict()
    pix_data = cv2.imread(img)
    width = pix_data.shape[1]
    height = pix_data.shape[0]

    file_pixel_dict[file_id_] = width * height

    for x in range(width):
        for y in range(height):
            cls_id = pix_data[y, x][0]
            dic_class_pixelcount[cls_id] = dic_class_pixelcount.get(cls_id, 0) + 1

            if x < symbol_size or x > (width-symbol_size) or y < symbol_size or y > (height-symbol_size):
                continue
            exist_file = False
            v = id_labeled_dict[cls_id]
            if isinstance(v, list):
                if len(v) == limit:
                    exist_file = True

                if not exist_file:
                    for v1 in v:
                        if file_id_ in v1:
                            exist_file = True
                            break

                if not exist_file:
                    new_v = {file_id_: (x, y)}
                    v.append(new_v)
                    id_labeled_dict[cls_id] = v

    for k, v in dic_class_pixelcount.items():
        id_pixel_dict[k] = {file_id_: v}

    return dic_class_pixelcount


def count_all_pixels(image_list):
    dic_class_pixelcount = dict()
    overall_pixelcount = dict()
    dic_class_imagecount = dict()
    result = dict()

    ret_count = 100
    ret_index = 0
    for img in image_list:
        ret_index += 1
        sys.stdout.write('.')
        if ret_index % ret_count == 0:
            sys.stdout.write('\n')
        sys.stdout.flush()

        for key, value in get_class_per_image(img).items():
            # Sum up the number of classes returned from get_class_per_image function
            overall_pixelcount[key] = overall_pixelcount.get(key, 0) + value

            # If the class is present in the image, then increase the value by one
            # shows in how many images a particular class is present
            key_file = str(img).split("/")
            key_file = key_file[len(key_file) - 1]
            total_pixel = file_pixel_dict[key_file]
            dic_class_pixelcount[key] = dic_class_pixelcount.get(key, 0) + total_pixel

            # id=>file_count
            dic_class_imagecount[key] = dic_class_imagecount.get(key, 0) + 1

    print("Done")
    # Save above 2 variables in a list
    for (k, v), (k2, v2), (k3, v3) in zip(overall_pixelcount.items(), dic_class_pixelcount.items(), dic_class_imagecount.items()):
        if k != k2:
            print ("This was impossible to happen, but somehow it did")
            exit()
        result[k] = [v, v2, v3]

    items = result.items()
    items.sort()
    for k, (v1, v2, v3) in items:
        name = id_name_dict[k]
        print('class name:%20s  id:%3s  fileCount:%5s  pixel:%5s  /%10s' % (name, str(k), str(v3), str(v1), str(v2)))
    return result


def cal_class_weights(image_list):
    freq_images = dict()
    weights = collections.OrderedDict()
    # calculate freq per class
    for k, (v1, v2, v3) in count_all_pixels(image_list).items():
        freq_images[k] = (v1 * 1.0) / (v2 * 1.0)
    # calculate median of freqs
    for k, v in freq_images.items():
        print(str(k) + ":" + str(v))

    print(freq_images.values())

    images_value = list(freq_images.values())

    median = np.median(images_value)
    # calculate weights
    for k, v in freq_images.items():
        weights[k] = median / v
    return weights

results = cal_class_weights(image_list)

for k, v in results.items():
    print("    class", k, "weight:", round(v, 4))

print("Copy this:")
for k, v in results.items():
    # print ("class_weighting: ", round(v, 4))
    print("class_weighting: {}".format(round(v, 4)))

# label symbol
for cls_id, file_pixel in id_labeled_dict.items():
    cls_name = id_name_dict[cls_id]
    if cls_id < 10:
        name_id = "0{}_{}".format(cls_id, cls_name)
    else:
        name_id = "{}_{}".format(cls_id, cls_name)

    cls_dir = os.path.join(class_dir, name_id)
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir)

    if isinstance(file_pixel, list):
        for dict_v in file_pixel:
            if isinstance(dict_v, dict):
                for k, v in dict_v.items():
                    file_id = k
                    pixel_coord = v

                    # file_id is both png...
                    src_file = os.path.join(image_dir, file_id)
                    if not os.path.exists(src_file):
                        file_id_list = str(file_id).split(".")
                        file_id = file_id_list[0]
                        file_id = file_id + ".jpg"
                        src_file = os.path.join(image_dir, file_id)
                    if not os.path.exists(src_file):
                        continue
                    file_store = os.path.join(cls_dir, file_id)

                    lineThickness = 3
                    src_img = cv2.imread(src_file)
                    v = list(v)
                    x = v[0]
                    y = v[1]

                    x1 = x - symbol_size
                    y1 = y2 = y
                    x2 = x + symbol_size
                    # horizontal
                    cv2.line(src_img, (x1, y1), (x2, y2), (0, 0, 255), lineThickness)

                    # vertical
                    x3 = x4 = x
                    y3 = y - symbol_size
                    y4 = y + symbol_size
                    cv2.line(src_img, (x3, y3), (x4, y4), (0, 0, 255), lineThickness)

                    cv2.imwrite(file_store, src_img)
