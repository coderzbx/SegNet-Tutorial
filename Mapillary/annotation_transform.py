# -*-coding:utf-8-*-

from PIL import Image
import cv2
import argparse
import os

import multiprocessing
from multiprocessing import Queue

# label_transform：分类合并的dictionary
# unlabeled取dictionary最后一个key
# src_dir：要转换的标注数据，dictionary的key对应的图片
# dest_dir：要保存转换后的标注数据，dictionary的value对应的图片

class Task:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file


class LabelTransform:
    def __init__(self):
        self.label_transform = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 2
        }
        self.queue = Queue()
        items = self.label_transform.items()
        items.sort()
        # last one is unlabeled id
        for k, v in items:
            self.unlabeled = k
        return

    def enter_queue(self, input_dir, output_dir):
        origin_list = os.listdir(input_dir)
        for _image in origin_list:
            image_path = os.path.join(input_dir, _image)
            result_path = os.path.join(output_dir, _image)
            name_list = _image.split('.')
            if len(name_list) < 2:
                print(image_path)
                continue

            ext_name = name_list[1]
            if ext_name != 'png' and ext_name != 'jpg':
                continue

            if os.path.exists(result_path):
                continue

            task = Task(image_path, result_path)
            self.queue.put(task)

    def transform(self):
        while not self.queue.empty():
            task = self.queue.get()
            image_path = task.input_file
            result_path = task.output_file

            print(image_path)

            img = cv2.imread(image_path)

            width = img.shape[1]
            height = img.shape[0]

            anna_img = Image.new('L', (width, height))

            img_data = anna_img.load()
            for x in range(width):
                for y in range(height):
                    cls_id = img[y, x][0]
                    if cls_id in self.label_transform:
                        label_id = self.label_transform[cls_id]
                        img_data[x, y] = label_id
                    else:
                        img_data[x, y] = self.unlabeled

            anna_img.save(result_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir

    transFormer = LabelTransform()

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    transFormer.enter_queue(src_dir, dest_dir)
    # transFormer.transform()

    process1 = multiprocessing.Process(target=transFormer.transform)
    process2 = multiprocessing.Process(target=transFormer.transform)
    process3 = multiprocessing.Process(target=transFormer.transform)
    process4 = multiprocessing.Process(target=transFormer.transform)

    process1.start()
    process2.start()
    process3.start()
    process4.start()