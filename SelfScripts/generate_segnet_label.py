from SelfScripts.segnet_label import segnet_lane_labels
from PIL import Image
import cv2
import argparse
import os

import multiprocessing


class LabelTransform:
    def __init__(self):
        self.label_clr = {l.id: l.color for l in segnet_lane_labels}
        return

    def transform(self, input_dir, output_dir):
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

    instance_dir = os.path.join(dir, "testannot")
    label_dir = os.path.join(dir, "testlabel")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # process1 = multiprocessing.Process(target=transFormer.transform,
    #                                   args=(instance_dir, label_dir))

    transFormer.transform(input_dir=instance_dir, output_dir=label_dir)

    instance_dir = os.path.join(dir, "valannot")
    label_dir = os.path.join(dir, "vallabel")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    transFormer.transform(input_dir=instance_dir, output_dir=label_dir)

    # process2 = multiprocessing.Process(target=transFormer.transform,
    #                                    args=(instance_dir, label_dir))

    instance_dir = os.path.join(dir, "trainannot")
    label_dir = os.path.join(dir, "trainlabel")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    transFormer.transform(input_dir=instance_dir, output_dir=label_dir)

    # process3 = multiprocessing.Process(target=transFormer.transform,
    #                                    args=(instance_dir, label_dir))

    # process1.start()
    # process2.start()
    # process3.start()