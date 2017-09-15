import os
import numpy as np
import cv2
from PIL import Image
import argparse
import caffe

import threading

from mapillary_label import mapillary_labels

IMAGE_DIR = '/data/deeplearning/dataset/mapillary'
thread_count = 16
cpu_count = 4

input_shape = [3, 3, 360, 480]


class FormatTrainSet:
    def __init__(self):
        self.train_limit = 0
        self.val_limit = 0
        self.dir = IMAGE_DIR

        self.threads = []

        self.clr_dict = {}
        for l in mapillary_labels:
            clr = l.color
            clr = clr[::-1]
            self.clr_dict[clr] = l.classId

        self.caffe_dir = self.dir + '_caffe'
        if not os.path.exists(self.caffe_dir):
            os.makedirs(self.caffe_dir)

        if not os.path.exists(self.caffe_dir + '/training'):
            os.makedirs(self.caffe_dir + '/training')

        if not os.path.exists(self.caffe_dir + '/validation'):
            os.makedirs(self.caffe_dir + '/validation')

        if not os.path.exists(self.caffe_dir + '/training/images'):
            os.makedirs(self.caffe_dir + '/training/images')

        if not os.path.exists(self.caffe_dir + '/validation/images'):
            os.makedirs(self.caffe_dir + '/validation/images')

        if not os.path.exists(self.caffe_dir + '/training/instances'):
            os.makedirs(self.caffe_dir + '/training/instances')

        if not os.path.exists(self.caffe_dir + '/validation/instances'):
            os.makedirs(self.caffe_dir + '/validation/instances')

        if not os.path.exists(self.caffe_dir + '/training/labels'):
            os.makedirs(self.caffe_dir + '/training/labels')

        if not os.path.exists(self.caffe_dir + '/validation/labels'):
            os.makedirs(self.caffe_dir + '/validation/labels')

    def convert_label(self, caffe_train_annot_dir, file_list, start_index, end_index, train_annot_dir):
        cur = start_index
        print(str(start_index)+"-"+str(end_index))
        while(cur < end_index):
            id_ = file_list[cur]
            id_ext = id_.split('.')
            id_ext = id_ext[1]
            if id_ext != 'png':
                continue

            if cur < start_index:
                cur += 1
                continue

            if cur == end_index:
                break

            # resize image and annotation
            image = os.path.join(caffe_train_annot_dir, id_)
            caffe_image = os.path.join(train_annot_dir, id_)

            img = cv2.imread(image)
            print(str(cur) + ":" + image)
            width = img.shape[1]
            height = img.shape[0]

            label_img = Image.new('L', (width, height))
            img_data = label_img.load()

            for i in range(width):
                for j in range(height):
                    color = img[j, i]
                    color = tuple(color)

                    if color in self.clr_dict:
                        label_num = self.clr_dict[color]
                        img_data[i, j] = label_num
            label_img.save(caffe_image)

            cur += 1

    def multi_format_label(self):
        caffe_train_annot_dir = os.path.join(self.caffe_dir, 'training/labels')
        train_annot_dir = os.path.join(self.caffe_dir, 'training/annotations')
        if not os.path.exists(train_annot_dir):
            os.makedirs(train_annot_dir)

        train_annot_files = os.listdir(caffe_train_annot_dir)
        count = len(train_annot_files)

        step = int(count / thread_count)

        for i in range(thread_count):
            start_index = step * i
            if i == thread_count - 1:
                end_index = count - 1
            else:
                end_index = step * (i + 1) - 1

            t1 = threading.Thread(target=self.convert_label,
                                  args=(caffe_train_annot_dir,
                                        train_annot_files, start_index, end_index,
                                        train_annot_dir))
            self.threads.append(t1)

        caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/labels')
        val_annot_dir = os.path.join(self.caffe_dir, 'validation/annotations')
        if not os.path.exists(val_annot_dir):
            os.makedirs(val_annot_dir)

        val_annot_files = os.listdir(caffe_val_annot_dir)
        count = len(val_annot_files)

        step = int(count / 2)

        for i in range(2):
            start_index = step * i
            if i == 2 - 1:
                end_index = count - 1
            else:
                end_index = step * (i + 1) - 1

            t1 = threading.Thread(target=self.convert_label,
                                  args=(caffe_val_annot_dir,
                                        val_annot_files, start_index, end_index,
                                        val_annot_dir))
            self.threads.append(t1)

    def format_instance(self):
        return
        caffe_train_annot_dir = os.path.join(self.caffe_dir, 'training/instances')
        train_annot_dir = os.path.join(self.caffe_dir, 'training/annotations')
        if not os.path.exists(train_annot_dir):
            os.makedirs(train_annot_dir)

        train_annot_files = os.listdir(caffe_train_annot_dir)
        for id_ in train_annot_files:
            id_ext = id_.split('.')
            id_ext = id_ext[1]
            if id_ext != 'png':
                continue
            # resize image and annotation
            image = os.path.join(caffe_train_annot_dir, id_)
            caffe_image = os.path.join(train_annot_dir, id_)

            # img = cv2.imread(image, 0)
            # a_img = np.array(img, np.double)
            # normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
            # cv2.imwrite(caffe_image, normalized)

            im = Image.open(image)
            table = [i / 256 for i in range(65536)]
            im2 = im.point(table, 'L')
            im2 = im2.convert('L')
            im2.save(caffe_image)

        caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/instances')
        val_annot_dir = os.path.join(self.caffe_dir, 'validation/annotations')
        if not os.path.exists(val_annot_dir):
            os.makedirs(val_annot_dir)

        val_annot_files = os.listdir(caffe_val_annot_dir)
        for id_ in val_annot_files:
            id_ext = id_.split('.')
            id_ext = id_ext[1]
            if id_ext != 'png':
                continue
            # resize image and annotation

            image = os.path.join(caffe_val_annot_dir, id_)
            caffe_image = os.path.join(val_annot_dir, id_)

            # img = cv2.imread(image, 0)
            # a_img = np.array(img, np.double)
            # normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
            # cv2.imwrite(caffe_image, normalized)

            im = Image.open(image)
            table = [i / 256 for i in range(65536)]
            im2 = im.point(table, 'L')
            im2 = im2.convert('L')
            im2.save(caffe_image)

    def format_label(self):
        caffe_train_annot_dir = os.path.join(self.caffe_dir, 'training/labels')
        train_annot_dir = os.path.join(self.caffe_dir, 'training/annotations')
        if not os.path.exists(train_annot_dir):
            os.makedirs(train_annot_dir)

        train_annot_files = os.listdir(caffe_train_annot_dir)
        for id_ in train_annot_files:
            id_ext = id_.split('.')
            id_ext = id_ext[1]
            if id_ext != 'png':
                continue
            # resize image and annotation
            image = os.path.join(caffe_train_annot_dir, id_)
            caffe_image = os.path.join(train_annot_dir, id_)

            # img = cv2.imread(image, 0)
            # a_img = np.array(img, np.double)
            # normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
            # cv2.imwrite(caffe_image, normalized)

            # im = Image.open(image)
            # table = [i / 256 for i in range(65536)]
            # im2 = im.point(table, 'L')
            # im2 = im.convert('L')
            # im2.save(caffe_image)

            img = cv2.imread(image)
            print(image)
            width = img.shape[1]
            height = img.shape[0]

            label_img = Image.new('L', (width, height))
            img_data = label_img.load()

            for i in range(width):
                for j in range(height):
                    color = img[j, i]
                    color = tuple(color)

                    if color in self.clr_dict:
                        label_num = self.clr_dict[color]
                        img_data[i, j] = label_num
            label_img.save(caffe_image)

        caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/labels')
        val_annot_dir = os.path.join(self.caffe_dir, 'validation/annotations')
        if not os.path.exists(val_annot_dir):
            os.makedirs(val_annot_dir)

        val_annot_files = os.listdir(caffe_val_annot_dir)
        for id_ in val_annot_files:
            id_ext = id_.split('.')
            id_ext = id_ext[1]
            if id_ext != 'png':
                continue
            # resize image and annotation

            image = os.path.join(caffe_val_annot_dir, id_)
            caffe_image = os.path.join(val_annot_dir, id_)

            # img = cv2.imread(image, 0)
            # a_img = np.array(img, np.double)
            # normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
            # cv2.imwrite(caffe_image, normalized)

            # im = Image.open(image)
            # table = [i / 256 for i in range(65536)]
            # im2 = im.point(table, 'L')
            # im2 = im.convert('L')
            # im2.save(caffe_image)

            img = cv2.imread(image)
            print(image)
            width = img.shape[1]
            height = img.shape[0]

            label_img = Image.new('L', (width, height))
            img_data = label_img.load()

            for i in range(width):
                for j in range(height):
                    color = img[j, i]
                    color = tuple(color)

                    if color in self.clr_dict:
                        label_num = self.clr_dict[color]
                        img_data[i, j] = label_num
            label_img.save(caffe_image)

    def resize(self, resize_image=False, resize_label=False, resize_instance=False):
        # training dataset
        train_list_file = os.path.join(self.dir, 'training')

        train_image_dir = os.path.join(train_list_file, 'images')
        caffe_train_image_dir = os.path.join(self.caffe_dir, 'training/images')

        if resize_image:
            train_image_files = os.listdir(train_image_dir)
            for id_ in train_image_files:
                id_ext = id_.split('.')
                id_ext = id_ext[1]
                if id_ext != 'jpg':
                    continue
                # resize image and annotation
                image = os.path.join(train_image_dir, id_)
                im1 = Image.open(image)
                im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

                caffe_image = os.path.join(caffe_train_image_dir, id_)
                im2.save(caffe_image)

        caffe_train_label_dir = os.path.join(self.caffe_dir, 'training/labels')
        train_label_dir = os.path.join(train_list_file, 'labels')

        if resize_label:
            train_label_files = os.listdir(train_label_dir)
            for id_ in train_label_files:
                id_ext = id_.split('.')
                id_ext = id_ext[1]
                if id_ext != 'png':
                    continue
                # resize image and annotation
                image = os.path.join(train_label_dir, id_)
                im1 = Image.open(image)
                im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

                caffe_image = os.path.join(caffe_train_label_dir, id_)
                im2.save(caffe_image)

        caffe_train_annot_dir = os.path.join(self.caffe_dir, 'training/instances')
        train_annot_dir = os.path.join(train_list_file, 'instances')

        if resize_instance:
            train_annot_files = os.listdir(train_annot_dir)
            for id_ in train_annot_files:
                id_ext = id_.split('.')
                id_ext = id_ext[1]
                if id_ext != 'png':
                    continue
                # resize image and annotation
                image = os.path.join(train_annot_dir, id_)
                im1 = Image.open(image)
                im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

                caffe_image = os.path.join(caffe_train_annot_dir, id_)
                im2.save(caffe_image)

        # validation
        val_list_file = os.path.join(self.dir, 'validation')
        val_image_dir = os.path.join(val_list_file, 'images')
        caffe_val_image_dir = os.path.join(self.caffe_dir, 'validation/images')

        if resize_image:
            val_image_files = os.listdir(val_image_dir)
            for id_ in val_image_files:
                id_ext = id_.split('.')
                id_ext = id_ext[1]
                if id_ext != 'jpg':
                    continue
                # resize image and annotation
                image = os.path.join(val_image_dir, id_)
                im1 = Image.open(image)
                im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

                caffe_image = os.path.join(caffe_val_image_dir, id_)
                im2.save(caffe_image)

        if resize_label:
            caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/labels')
            val_annot_dir = os.path.join(val_list_file, 'labels')
            val_annot_files = os.listdir(val_annot_dir)
            for id_ in val_annot_files:
                id_ext = id_.split('.')
                id_ext = id_ext[1]
                if id_ext != 'png':
                    continue
                # resize image and annotation
                image = os.path.join(val_annot_dir, id_)
                im1 = Image.open(image)
                im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

                caffe_image = os.path.join(caffe_val_annot_dir, id_)
                im2.save(caffe_image)

        if resize_instance:
            caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/instances')
            val_annot_dir = os.path.join(val_list_file, 'instances')
            val_annot_files = os.listdir(val_annot_dir)
            for id_ in val_annot_files:
                id_ext = id_.split('.')
                id_ext = id_ext[1]
                if id_ext != 'png':
                    continue
                # resize image and annotation
                image = os.path.join(val_annot_dir, id_)
                im1 = Image.open(image)
                im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

                caffe_image = os.path.join(caffe_val_annot_dir, id_)
                im2.save(caffe_image)

    def format(self):
        train_list_file = os.path.join(self.caffe_dir, 'training')
        train_txt = '/SegNet/CamVid/train.txt'
        val_list_file = os.path.join(self.caffe_dir, 'validation')
        val_txt = '/SegNet/CamVid/val.txt'

        # foreach files
        train_image_dir = os.path.join(train_list_file, 'images')
        train_annot_dir = os.path.join(train_list_file, 'annotations')

        train_image_files = os.listdir(train_image_dir)

        train_images = []
        train_annots = []
        index = 0
        for id_ in train_image_files:
            file_name = id_.split('.')
            file_ex = file_name[1]
            if file_ex != 'png' and file_ex != 'jpg':
                continue
            file_name = file_name[0]

            train_images.append(os.path.join(train_image_dir, '{}.jpg'.format(file_name)))
            train_annots.append(os.path.join(train_annot_dir, '{}.png'.format(file_name)))
            index += 1
            if index == self.train_limit:
                break

        # train_images = [os.path.join(train_image_dir, id_) for id_ in train_image_files]
        # train_annot_files = os.listdir(train_annot_dir)
        # train_annots = [os.path.join(train_annot_dir, id_) for id_ in train_annot_files]

        image_count = len(train_images)
        label_count = len(train_annots)
        with open(train_txt, 'wb') as f:
            if image_count == label_count:
                for image, annot in zip(train_images, train_annots):
                    str = image + ' ' + annot + '\n'
                    f.write(str)

        val_image_dir = os.path.join(val_list_file, 'images')
        val_annot_dir = os.path.join(val_list_file, 'annotations')

        val_images = []
        val_annots = []
        val_image_files = os.listdir(val_image_dir)
        index = 0
        for id_ in val_image_files:
            file_name = id_.split('.')
            file_ex = file_name[1]
            if file_ex != 'png' and file_ex != 'jpg':
                continue
            file_name = file_name[0]
            val_images.append(os.path.join(val_image_dir, '{}.jpg'.format(file_name)))
            val_annots.append(os.path.join(val_annot_dir, '{}.png'.format(file_name)))

            index += 1
            if index == self.val_limit:
                break

        # val_images = [os.path.join(val_image_dir, id_) for id_ in val_image_files]
        # val_annot_files = os.listdir(val_annot_dir)
        # val_annots = [os.path.join(val_annot_dir, id_) for id_ in val_annot_files]

        image_count = len(val_images)
        label_count = len(val_annots)
        with open(val_txt, 'wb') as f:
            if image_count == label_count:
                for image, annot in zip(val_images, val_annots):
                    str = image + ' ' + annot + '\n'
                    f.write(str)


if __name__ == '__main__':
    handle = FormatTrainSet()
    # if needed to resize
    resize_image = False
    resize_label = False
    resize_instance = False
    handle.resize(resize_image=resize_image, resize_label=resize_label, resize_instance=resize_instance)
    # if needed to format label to 0-255
    # handle.multi_format_label()
    # handle.format_label()
    handle.format()

    # for t in handle.threads:
    #     t.setDaemon(True)
    #     t.start()
    #
    # for t in handle.threads:
    #     t.join()