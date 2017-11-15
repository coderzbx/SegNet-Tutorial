# -*-coding:utf-8-*-

import numpy as np
import argparse
import os
from os import listdir
import sys
import collections
import cv2

from Mapillary.mapillary_label import mapillary_instance_labels
from multiprocessing import Queue


class SingleImage:
    def __init__(self, dic_class_pixelcount):
        self.dic_class_pixelcount = dic_class_pixelcount

class SumTask:
    def __init__(self, dic_class_pixelcount, stop_flag=False):
        self.dic_class_pixelcount = dic_class_pixelcount
        self.stop_flag = stop_flag


class CalculateWeight:
    def __init__(self, label_dir, image_dir, class_dir, limit):
        self.symbol_size = 50
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.class_dir = class_dir
        self.limit = limit

        self.overall_pixelcount = dict()
        self.dic_class_pixelcount = dict()
        self.single_queue = Queue()
        self.sum_queue = Queue()
        self.calc_queue = Queue()

        # id->class_name
        self.id_name_dict = {l.id: l.name for l in mapillary_instance_labels}

        # id=>[{filename=>pixel_coord}],limit
        self.id_labeled_dict = {l.id: [] for l in mapillary_instance_labels}

        # file_name=>total_pixel_count
        self.file_pixel_dict = {}

        # id=>{file_name=>pixel_count}
        self.id_pixel_dict = {l.id: {} for l in mapillary_instance_labels}

        # id=>file_count
        self.id_file_count = {l.id: 0 for l in mapillary_instance_labels}


    def start_queue(self):
        return

    def add_sum(self):
        while True:
            return

    def get_class_per_image(self, img):
        name_list = str(img).split("/")
        file_id_ = name_list[len(name_list) - 1]

        dic_class_pixelcount = dict()
        pix_data = cv2.imread(img)
        width = pix_data.shape[1]
        height = pix_data.shape[0]

        self.file_pixel_dict[file_id_] = width * height

        for x in range(width):
            for y in range(height):
                cls_id = pix_data[y, x][0]
                dic_class_pixelcount[cls_id] = dic_class_pixelcount.get(cls_id, 0) + 1

                if x < self.symbol_size or x > (width - self.symbol_size) or y < self.symbol_size or y > (height - self.symbol_size):
                    continue

                exist_file = False
                v = self.id_labeled_dict[cls_id]
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
                        self.id_labeled_dict[cls_id] = v

        for k, v in dic_class_pixelcount.items():
            self.id_pixel_dict[k] = {file_id_: v}

        return dic_class_pixelcount

    def count_all_pixels(self, image_list):
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

            for key, value in self.get_class_per_image(img).items():
                # Sum up the number of classes returned from get_class_per_image function
                overall_pixelcount[key] = overall_pixelcount.get(key, 0) + value

                # If the class is present in the image, then increase the value by one
                # shows in how many images a particular class is present
                key_file = str(img).split("/")
                key_file = key_file[len(key_file) - 1]
                total_pixel = self.file_pixel_dict[key_file]
                dic_class_pixelcount[key] = dic_class_pixelcount.get(key, 0) + total_pixel

                # id=>file_count
                dic_class_imagecount[key] = dic_class_imagecount.get(key, 0) + 1

        print("Done")
        # Save above 2 variables in a list
        for (k, v), (k2, v2), (k3, v3) in zip(overall_pixelcount.items(), dic_class_pixelcount.items(),
                                              dic_class_imagecount.items()):
            if k != k2:
                print ("This was impossible to happen, but somehow it did")
                exit()
            result[k] = [v, v2, v3]

        items = result.items()
        items.sort()
        for k, (v1, v2, v3) in items:
            name = self.id_name_dict[k]
            print('class name:%20s  id:%3s  fileCount:%5s  pixel:%5s  /%10s' % (name, str(k), str(v3), str(v1), str(v2)))
        return result

    def cal_class_weights(self, image_list):
        freq_images = dict()
        weights = collections.OrderedDict()
        # calculate freq per class
        for k, (v1, v2, v3) in self.count_all_pixels(image_list).items():
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

    def label_symbol(self):
        for cls_id, file_pixel in self.id_labeled_dict.items():
            cls_name = self.id_name_dict[cls_id]
            if cls_id < 10:
                name_id = "0{}_{}".format(cls_id, cls_name)
            else:
                name_id = "{}_{}".format(cls_id, cls_name)

            cls_dir = os.path.join(self.class_dir, name_id)
            if not os.path.exists(cls_dir):
                os.makedirs(cls_dir)

            if isinstance(file_pixel, list):
                for dict_v in file_pixel:
                    if isinstance(dict_v, dict):
                        for k, v in dict_v.items():
                            file_id = k

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

                            line_thickness = 3
                            src_img = cv2.imread(src_file)
                            v = list(v)
                            x = v[0]
                            y = v[1]

                            x1 = x - self.symbol_size
                            y1 = y2 = y
                            x2 = x + self.symbol_size
                            # horizontal
                            cv2.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

                            # vertical
                            x3 = x4 = x
                            y3 = y - self.symbol_size
                            y4 = y + self.symbol_size
                            cv2.line(src_img, (x3, y3), (x4, y4), (0, 255, 0), line_thickness)

                            cv2.imwrite(file_store, src_img)


if __name__ == '__main__':
    # Import arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, help='Path to the folder containing the images with annotations')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--class_dir', type=str)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    label_dir = args.label_dir
    class_dir = args.class_dir
    limit = args.limit
    if limit <= 0:
        limit = 5
    image_dir = args.image_dir

    images = listdir(label_dir)
    # Keep only images and append image_names to directory
    image_list = [os.path.join(label_dir, s) for s in images if s.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("Number of images:%d" % len(image_list))

    calculate_proc = CalculateWeight(label_dir, image_dir, class_dir, limit)
    results = calculate_proc.cal_class_weights(image_list)

    for k, v in results.items():
        print("    class", k, "weight:", round(v, 4))

    print("Copy this:")
    for k, v in results.items():
        # print ("class_weighting: ", round(v, 4))
        print("class_weighting: {}".format(round(v, 4)))

    # label symbol
    calculate_proc.label_symbol()
