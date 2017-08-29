import os
import numpy as np
import cv2
from PIL import Image
import argparse
import caffe

IMAGE_DIR = '/data/deeplearning/dataset/mapillary'

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
args = parser.parse_args()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

input_shape = net.blobs['data'].data.shape


class FormatTrainSet:
    def __init__(self):
        self.train_limit = 0
        self.val_limit = 0
        self.dir = IMAGE_DIR
        self.caffe_dir = self.dir + '_caffe'
        if not os.path.exists(self.caffe_dir):
            os.makedirs(self.caffe_dir)
            os.makedirs(self.caffe_dir + '/training')
            os.makedirs(self.caffe_dir + '/validation')
            os.makedirs(self.caffe_dir + '/training/images')
            os.makedirs(self.caffe_dir + '/validation/images')
            os.makedirs(self.caffe_dir + '/training/instances')
            os.makedirs(self.caffe_dir + '/validation/instances')

    def format_label(self):
        caffe_train_annot_dir = os.path.join(self.caffe_dir, 'training/instances')
        train_annot_dir = os.path.join(self.caffe_dir, 'training/annotations')
        if not os.path.exists(train_annot_dir):
            os.makedirs(train_annot_dir)

        train_annot_files = os.listdir(caffe_train_annot_dir)
        for id_ in train_annot_files:
            # resize image and annotation
            image = os.path.join(caffe_train_annot_dir, id_)
            img = cv2.imread(image, 0)
            a_img = np.array(img, np.double)
            normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
            caffe_image = os.path.join(train_annot_dir, id_)
            cv2.imwrite(caffe_image, normalized)

        caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/instances')
        val_annot_dir = os.path.join(self.caffe_dir, 'validation/annotations')
        if not os.path.exists(val_annot_dir):
            os.makedirs(val_annot_dir)

        val_annot_files = os.listdir(caffe_val_annot_dir)
        for id_ in val_annot_files:
            # resize image and annotation
            image = os.path.join(caffe_val_annot_dir, id_)
            img = cv2.imread(image, 0)
            a_img = np.array(img, np.double)
            normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
            caffe_image = os.path.join(val_annot_dir, id_)
            cv2.imwrite(caffe_image, normalized)

    def resize(self):
        # training dataset
        train_list_file = os.path.join(self.dir, 'training')

        train_image_dir = os.path.join(train_list_file, 'images')
        caffe_train_image_dir = os.path.join(self.caffe_dir, 'training/images')
        train_image_files = os.listdir(train_image_dir)
        train_image_files = []

        for id_ in train_image_files:
            # resize image and annotation
            image = os.path.join(train_image_dir, id_)
            im1 = Image.open(image)
            im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

            caffe_image = os.path.join(caffe_train_image_dir, id_)
            im2.save(caffe_image)

        caffe_train_annot_dir = os.path.join(self.caffe_dir, 'training/instances')
        train_annot_dir = os.path.join(train_list_file, 'instances')
        train_annot_files = os.listdir(train_annot_dir)
        for id_ in train_annot_files:
            # resize image and annotation
            image = os.path.join(train_annot_dir, id_)
            im1 = Image.open(image).convert('L')
            im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)
            im2 = im2.convert('L')

            caffe_image = os.path.join(caffe_train_annot_dir, id_)
            im2.save(caffe_image)

        # validation
        val_list_file = os.path.join(self.dir, 'validation')
        val_image_dir = os.path.join(val_list_file, 'images')
        caffe_val_image_dir = os.path.join(self.caffe_dir, 'validation/images')
        val_image_files = os.listdir(val_image_dir)
        val_image_files = []

        for id_ in val_image_files:
            # resize image and annotation
            image = os.path.join(val_image_dir, id_)
            im1 = Image.open(image)
            im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)

            caffe_image = os.path.join(caffe_val_image_dir, id_)
            im2.save(caffe_image)

        caffe_val_annot_dir = os.path.join(self.caffe_dir, 'validation/instances')
        val_annot_dir = os.path.join(val_list_file, 'instances')
        val_annot_files = os.listdir(val_annot_dir)
        for id_ in val_annot_files:
            # resize image and annotation
            image = os.path.join(val_annot_dir, id_)
            im1 = Image.open(image).convert('L')
            im2 = im1.resize((input_shape[3], input_shape[2]), Image.NEAREST)
            im2 = im2.convert('L')

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
    # handle.resize()
    # if needed to format label to 0-255
    # handle.format_label()
    handle.format()