import os
import cv2
import argparse

import multiprocessing
from multiprocessing import Queue

cpu_count = 4
thread_count = 16
input_shape = [3, 3, 360, 480]


class ResizeTask:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file


class FormatTrainSet:
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.queue = Queue()

        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

        if not os.path.exists(self.dest_dir + 'images'):
            os.makedirs(self.dest_dir + 'images')

        if not os.path.exists(self.dest_dir + 'labels'):
            os.makedirs(self.dest_dir + 'labels')

        if not os.path.exists(self.dest_dir + 'annotations'):
            os.makedirs(self.dest_dir + 'annotations')

    def list_dir(self, src_image_dir, dest_image_dir):
        image_files = os.listdir(src_image_dir)
        for id_ in image_files:
            id_ext = id_.split('.')
            id_ext = id_ext[1]
            if id_ext != 'jpg' and id_ext != 'png':
                continue
            src_image = os.path.join(src_image_dir, id_)
            dest_image = os.path.join(dest_image_dir, id_)

            task = ResizeTask(src_image, dest_image)
            self.queue.put(task)

    def enter_queue(self, resize_image=False, resize_label=False, resize_instance=False):
        if resize_image:
            src_image_dir = os.path.join(self.src_dir, 'images')
            dest_image_dir = os.path.join(self.dest_dir, 'images')
            self.list_dir(src_image_dir, dest_image_dir)
        if resize_label:
            src_image_dir = os.path.join(self.src_dir, 'labels')
            dest_image_dir = os.path.join(self.dest_dir, 'labels')
            self.list_dir(src_image_dir, dest_image_dir)
        if resize_instance:
            src_image_dir = os.path.join(self.src_dir, 'annotations')
            dest_image_dir = os.path.join(self.dest_dir, 'annotations')
            self.list_dir(src_image_dir, dest_image_dir)

    def resize(self):
        while not self.queue.empty():
            task = self.queue.get()

            im1 = cv2.imread(task.input_file)
            im2 = cv2.resize(im1, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(task.output_file, im2)

    def format(self, image_dir, label_dir, txt_dir, image_type):
        train_image_files = os.listdir(image_dir)

        images = []
        annots = []

        for id_ in train_image_files:
            file_name = id_.split('.')
            file_ex = file_name[1]
            if file_ex != 'png' and file_ex != 'jpg':
                continue
            file_name = file_name[0]

            images.append(os.path.join(image_dir, '{}.{}'.format(file_name, file_ex)))
            annots.append(os.path.join(label_dir, '{}_L.png'.format(file_name)))

        images.sort()
        annots.sort()
        image_count = len(images)
        label_count = len(annots)

        train_txt = os.path.join(txt_dir, '{}.txt'.format(image_type))
        with open(train_txt, 'wb') as f:
            if image_count == label_count:
                for image, annot in zip(images, annots):
                    str = image + ' ' + annot + '\n'
                    f.write(str.encode("UTF-8"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=False)
    parser.add_argument('--dest_dir', type=str, required=False)
    args = parser.parse_args()

    src_dir = ''
    if args.src_dir and args.src_dir != '' and os.path.exists(args.src_dir):
        src_dir = args.src_dir
        print(src_dir)

    if src_dir:
        if not src_dir.endswith('/'):
            src_dir = src_dir + '/'

    dest_dir = ''
    if args.dest_dir and args.dest_dir != '' and os.path.exists(args.dest_dir):
        dest_dir = args.dest_dir
        print(dest_dir)

    if dest_dir:
        if not dest_dir.endswith('/'):
            dest_dir = dest_dir + '/'

    handle = FormatTrainSet(src_dir=src_dir, dest_dir=dest_dir)
    # if needed to resize
    resize_image = False
    resize_label = False
    resize_instance = False
    handle.enter_queue(resize_image=resize_image, resize_label=resize_label, resize_instance=resize_instance)
    handle.resize()

    type_list = ['train', 'val', 'test']
    for image_type in type_list:
        image_dir = os.path.join(dest_dir, image_type)
        label_dir = os.path.join(dest_dir, '{}annot'.format(image_type))
        handle.format(image_dir=image_dir, label_dir=label_dir, txt_dir=dest_dir, image_type=image_type)

