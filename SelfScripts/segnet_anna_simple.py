from SelfScripts.segnet_label import segnet_format_labels
from SelfScripts.segnet_label import segnet_labels
from PIL import Image
import cv2
import argparse
import os

import multiprocessing
from multiprocessing import Queue


class Task:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file


class LabelTransform:
    def __init__(self):
        self.clr_label = {l.color: l.categoryId for l in segnet_labels}
        self.queue = Queue()
        self.unlabeled = 2
        self.other_label = 0
        self.choose_label = 11
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
                    color = img[y, x]
                    color = color[::-1]
                    color = tuple(color)
                    if color in self.clr_label:
                        label_id = self.clr_label[color]
                        if label_id == self.choose_label:
                            img_data[x, y] = 1
                        elif label_id == 12:
                            img_data[x, y] = self.unlabeled
                        else:
                            img_data[x, y] = self.other_label
                    else:
                        img_data[x, y] = self.unlabeled

            anna_img.save(result_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=False)
    args = parser.parse_args()

    dir = ''
    if args.dir and args.dir != '' and os.path.exists(args.dir):
        dir = args.dir
        print(dir)

    if dir:
        if not dir.endswith('/'):
            dir = dir + '/'

    transFormer = LabelTransform()

    type_list = ["train", "val", "test"]
    for image_type in type_list:
        instance_dir = os.path.join(dir, "{}label".format(image_type))
        label_dir = os.path.join(dir, "{}annot".format(image_type))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        transFormer.enter_queue(instance_dir, label_dir)
    # transFormer.transform()

    process1 = multiprocessing.Process(target=transFormer.transform)
    process2 = multiprocessing.Process(target=transFormer.transform)
    process3 = multiprocessing.Process(target=transFormer.transform)
    process4 = multiprocessing.Process(target=transFormer.transform)

    process1.start()
    process2.start()
    process3.start()
    process4.start()