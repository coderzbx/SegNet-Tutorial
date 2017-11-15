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
from multiprocessing import Manager
import time
import multiprocessing


class SingleImage:
    def __init__(self, dic_class_pixelcount):
        self.dic_class_pixelcount = dic_class_pixelcount


class SumTask:
    def __init__(self, dic_class_pixelcount, image_file, stop_flag=False):
        self.dic_class_pixelcount = dic_class_pixelcount
        self.image_file = image_file
        self.stop_flag = stop_flag


class CalcTask:
    def __init__(self, count_all_pixels):
        self.count_all_pixels = count_all_pixels


class CalculateWeight:
    def __init__(self, label_dir, image_dir, class_dir, limit, mgr, total_count):
        self.total_count = total_count
        self.symbol_size = 50
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.class_dir = class_dir
        self.limit = limit

        self.single_queue = Queue()
        self.sum_queue = Queue()
        self.calc_queue = Queue()

        self.lock = multiprocessing.Lock()

        self.ret_index = 0

        self.mgr = mgr

        self.all_pixels = self.mgr.dict()
        self.dic_class_pixelcount = self.mgr.dict()
        self.overall_pixelcount = self.mgr.dict()
        self.dic_class_imagecount = self.mgr.dict()

        # id->class_name
        self.id_name_dict = {l.id: l.name for l in mapillary_instance_labels}

        # id=>[{filename=>pixel_coord}],limit
        # self.id_labeled_dict = {l.id: [] for l in mapillary_instance_labels}
        self.id_labeled_dict = self.mgr.dict()

        # file_name=>total_pixel_count
        # self.file_pixel_dict = dict()
        self.file_pixel_dict = self.mgr.dict()

        # id=>{file_name=>pixel_count}
        # self.id_pixel_dict = {l.id: {} for l in mapillary_instance_labels}
        self.id_pixel_dict = self.mgr.dict()

        # id=>file_count
        # self.id_file_count = {l.id: 0 for l in mapillary_instance_labels}
        self.id_file_count = self.mgr.dict()

    def start_queue(self, image_list):
        if not isinstance(image_list, list):
            return

        for _image in image_list:
            self.single_queue.put(_image)
            # name_list = str(_image).split("/")
            # file_id_ = name_list[len(name_list) - 1]
            #
            # pix_data = cv2.imread(_image)
            # width = pix_data.shape[1]
            # height = pix_data.shape[0]
            #
            # self.file_pixel_dict[file_id_] = width * height

        return

    def get_class_per_image(self, lock):
        while not self.single_queue.empty():
            # ret_count = 2
            # self.ret_index += 1
            # sys.stdout.write('.')
            # if self.ret_index % ret_count == 0:
            #     sys.stdout.write(str(self.ret_index))
            #     sys.stdout.write('\n')
            # sys.stdout.flush()

            _id_labeled_dict = dict()

            start = time.time()

            img = self.single_queue.get()

            name_list = str(img).split("/")
            file_id_ = name_list[len(name_list) - 1]

            dic_class_pixelcount = dict()
            pix_data = cv2.imread(img)
            width = pix_data.shape[1]
            height = pix_data.shape[0]

            for x in range(width):
                for y in range(height):
                    cls_id = pix_data[y, x][0]
                    dic_class_pixelcount[cls_id] = dic_class_pixelcount.get(cls_id, 0) + 1

                    if x < self.symbol_size or x > (width - self.symbol_size) or y < self.symbol_size or y > (
                        height - self.symbol_size):
                        continue

                    if cls_id not in _id_labeled_dict:
                        new_v = {file_id_: (x, y)}
                        _id_labeled_dict[cls_id] = new_v

            with lock:
                self.file_pixel_dict[file_id_] = width * height
                for k, v in _id_labeled_dict.items():
                    _v1 = []
                    if k in self.id_labeled_dict:
                        _v1 = self.id_labeled_dict[k]

                    exist_file = False
                    if isinstance(_v1, list):
                        if len(_v1) == limit:
                            exist_file = True

                        if not exist_file:
                            for v1 in _v1:
                                if file_id_ in v1:
                                    exist_file = True
                                    break

                        if not exist_file:
                            _v1.append(v)
                            self.id_labeled_dict[k] = _v1
                #
                # print("self.id_labeled_dict\n")
                # print(self.id_labeled_dict)
                # print("\n")

                for k, v in dic_class_pixelcount.items():
                    self.id_pixel_dict[k] = {file_id_: v}
                # print("self.id_pixel_dict\n")
                # print(self.id_pixel_dict)
                # print("\n")

            end = time.time()
            print("{} in {} s".format(img, (end-start)))

            next_task = SumTask(dic_class_pixelcount=dic_class_pixelcount, image_file=img)
            self.sum_queue.put(next_task)

        # 放入结束标志
        # next_task = SumTask(dic_class_pixelcount=None, image_file='', stop_flag=True)
        # self.sum_queue.put(next_task)

    def add_sum(self, lock):
        while True:
            if self.sum_queue.empty():
                time.sleep(1)
                continue

            task = self.sum_queue.get()
            if not isinstance(task, SumTask):
                break

            # if task.stop_flag:
            #     break

            dic_class_pixelcount = task.dic_class_pixelcount
            img = task.image_file
            self.ret_index += 1

            print(img)
            if not isinstance(dic_class_pixelcount, dict):
                break

            for key, value in dic_class_pixelcount.items():
                with lock:
                    # Sum up the number of classes returned from get_class_per_image function
                    self.overall_pixelcount[key] = self.overall_pixelcount.get(key, 0) + value

                    # If the class is present in the image, then increase the value by one
                    # shows in how many images a particular class is present
                    key_file = str(img).split("/")
                    key_file = key_file[len(key_file) - 1]
                    total_pixel = self.file_pixel_dict[key_file]
                    self.dic_class_pixelcount[key] = self.dic_class_pixelcount.get(key, 0) + total_pixel

                    # id=>file_count
                    self.dic_class_imagecount[key] = self.dic_class_imagecount.get(key, 0) + 1

            if self.ret_index == self.total_count:
                break

    def cal_class_weights(self):
        # Save above 2 variables in a list
        result = dict()

        overall_items = self.overall_pixelcount.items()
        overall_items.sort()
        for k, v in overall_items:
            v2 = self.dic_class_pixelcount[k]
            v3 = self.dic_class_imagecount[k]
            result[k] = [v, v2, v3]
        # for (k, v), (k2, v2), (k3, v3) in zip(overall_pixelcount.items(), dic_class_pixelcount.items(),
        #                                       dic_class_imagecount.items()):
        #     if k != k2:
        #         print ("This was impossible to happen, but somehow it did")
        #         exit()
        #     result[k] = [v, v2, v3]

        items = result.items()
        items.sort()
        for k, (v1, v2, v3) in items:
            name = self.id_name_dict[k]
            print('class name:%20s  id:%3s  fileCount:%5s  pixel:%5s  /%10s' % (name, str(k), str(v3), str(v1), str(v2)))

        self.all_pixels = result

        freq_images = dict()
        weights = collections.OrderedDict()
        # calculate freq per class
        for k, (v1, v2, v3) in self.all_pixels.items():
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

        for k, v in weights.items():
            print("    class", k, "weight:", round(v, 4))

        print("Copy this:")
        for k, v in weights.items():
            # print ("class_weighting: ", round(v, 4))
            print("class_weighting: {}".format(round(v, 4)))

    def label_symbol(self):
        print("id_labeled_dict:{}".format(len(self.id_labeled_dict)))
        print(self.id_labeled_dict)
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
                            cv2.line(src_img, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)

                            # vertical
                            x3 = x4 = x
                            y3 = y - self.symbol_size
                            y4 = y + self.symbol_size
                            cv2.line(src_img, (x3, y3), (x4, y4), (0, 0, 255), line_thickness)

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
    # images = [images[i] for i in range(0, limit)]
    # Keep only images and append image_names to directory
    image_list = [os.path.join(label_dir, s) for s in images if s.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("Number of images:%d" % len(image_list))

    mgr = multiprocessing.Manager()

    calculate_proc = CalculateWeight(label_dir, image_dir, class_dir, limit, mgr, len(images))
    calculate_proc.start_queue(image_list)

    print("waiting for insert images to queue....\n")
    time.sleep(5)

    lock = multiprocessing.Lock()

    process_per_image_1 = multiprocessing.Process(target=calculate_proc.get_class_per_image, args=(lock,))
    process_per_image_2 = multiprocessing.Process(target=calculate_proc.get_class_per_image, args=(lock,))
    process_per_image_3 = multiprocessing.Process(target=calculate_proc.get_class_per_image, args=(lock,))
    process_per_image_4 = multiprocessing.Process(target=calculate_proc.get_class_per_image, args=(lock,))

    process_per_image_1.start()
    process_per_image_2.start()
    process_per_image_3.start()
    process_per_image_4.start()

    print("start processing....\n")
    # calculate_proc.get_class_per_image(lock=lock)
    # calculate_proc.add_sum()

    # process_sum_proc = multiprocessing.Process(target=calculate_proc.add_sum)
    # process_sum_proc.start()

    process_per_image_1.join()
    process_per_image_2.join()
    process_per_image_3.join()
    process_per_image_4.join()

    # process_sum_proc.join()

    calculate_proc.add_sum(lock=lock)

    # label symbol
    print("export class_weighting....\n")
    calculate_proc.cal_class_weights()

    print("start processing label symbol by coordinate in pixel....\n")
    calculate_proc.label_symbol()

    print("finish\n")
