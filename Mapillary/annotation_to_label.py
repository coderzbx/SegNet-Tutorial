from SelfScripts.segnet_label import segnet_14_labels
from PIL import Image
import cv2
import argparse
import os

import multiprocessing
from multiprocessing import Queue

# label_clr：class_id对应的颜色


class Task:
    def __init__(self, src_file, dest_file):
        self.src_file = src_file
        self.dest_file = dest_file


class LabelTransform:
    def __init__(self):
        self.label_clr = {l.id: l.color for l in segnet_14_labels}
        self.queue = Queue()

    def start_queue(self, instance_dir, label_dir):
        origin_list = os.listdir(instance_dir)
        for _image in origin_list:
            image_path = os.path.join(instance_dir, _image)
            result_path = os.path.join(label_dir, _image)

            task = Task(image_path, result_path)
            self.queue.put(task)

    def transform(self):
        while not self.queue.empty():
            task = self.queue.get()

            image_path = task.src_file
            result_path = task.dest_file

            _image = str(image_path).split("/")
            _image = _image[len(_image) - 1]
            name_list = _image.split('.')
            if len(name_list) < 2:
                print(image_path)
                continue

            ext_name = name_list[1]
            if ext_name != 'png' and ext_name != 'jpg':
                continue

            print(image_path)

            img = cv2.imread(image_path)

            width = img.shape[1]
            height = img.shape[0]

            label_img = Image.new('RGB', (width, height), color=(255, 255, 255))
            img_data = label_img.load()
            for x in range(width):
                for y in range(height):
                    label_id = img[y, x][0]
                    color = self.label_clr[label_id]
                    img_data[x, y] = color

            label_img.save(result_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    args = parser.parse_args()

    instance_dir = args.annot_dir
    label_dir = args.label_dir

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    transFormer = LabelTransform()
    transFormer.start_queue(instance_dir, label_dir)

    process1 = multiprocessing.Process(target=transFormer.transform)
    process2 = multiprocessing.Process(target=transFormer.transform)
    process3 = multiprocessing.Process(target=transFormer.transform)
    process4 = multiprocessing.Process(target=transFormer.transform)

    process1.start()
    process2.start()
    process3.start()
    process4.start()