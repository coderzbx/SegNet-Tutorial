import os
import argparse
import shutil


class SubDataSet:
    def __init__(self, image_list, image_dir, label_dir, annot_dir, dest_dir):
        self.image_list = list(image_list)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.annot_dir = annot_dir
        self.dest_dir = dest_dir

    def run(self):
        if not isinstance(self.image_list, list):
            return

        for _image in self.image_list:
            name_list = str(_image).split("/")
            file_id = name_list[len(name_list) - 1]
            name_ext = str(file_id).split(".")
            file_name = name_ext[0]

            if os.path.exists(self.image_dir):
                dest_image_dir = os.path.join(self.dest_dir, "images")
                if not os.path.exists(dest_image_dir):
                    os.makedirs(dest_image_dir)
                src_image_path = os.path.join(self.image_dir, file_id)
                dest_image_path = os.path.join(dest_image_dir, file_id)
                if os.path.exists(src_image_path):
                    shutil.copyfile(src_image_path, dest_image_path)
                else:
                    print("file not exist [{}]\n".format(src_image_path))

            if os.path.exists(self.label_dir):
                dest_image_dir = os.path.join(self.dest_dir, "labels")
                if not os.path.exists(dest_image_dir):
                    os.makedirs(dest_image_dir)

                id_ = "{}.png".format(file_name)
                src_image_path = os.path.join(self.label_dir, id_)
                dest_image_path = os.path.join(dest_image_dir, id_)
                if os.path.exists(src_image_path):
                    shutil.copyfile(src_image_path, dest_image_path)
                else:
                    id_ = "{}.jpg".format(file_name)
                    src_image_path = os.path.join(self.label_dir, id_)
                    dest_image_path = os.path.join(dest_image_dir, id_)
                    if os.path.exists(src_image_path):
                        shutil.copyfile(src_image_path, dest_image_path)
                    else:
                        print("file not exist [{}]\n".format(src_image_path))

            if os.path.exists(self.annot_dir):
                dest_image_dir = os.path.join(self.dest_dir, "instances")
                if not os.path.exists(dest_image_dir):
                    os.makedirs(dest_image_dir)

                id_ = "{}.png".format(file_name)
                src_image_path = os.path.join(self.label_dir, id_)
                dest_image_path = os.path.join(dest_image_dir, id_)
                if os.path.exists(src_image_path):
                    shutil.copyfile(src_image_path, dest_image_path)
                else:
                    id_ = "{}.jpg".format(file_name)
                    src_image_path = os.path.join(self.label_dir, id_)
                    dest_image_path = os.path.join(dest_image_dir, id_)
                    if os.path.exists(src_image_path):
                        shutil.copyfile(src_image_path, dest_image_path)
                    else:
                        print("file not exist [{}]\n".format(src_image_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=False)
    parser.add_argument('--image_dir', type=str, required=False)
    parser.add_argument('--label_dir', type=str, required=False)
    parser.add_argument('--annot_dir', type=str, required=False)
    parser.add_argument('--dest_dir', type=str, required=False)
    args = parser.parse_args()

    src_dir = args.src_dir
    image_dir = args.image_dir
    label_dir = args.label_dir
    annot_dir = args.annot_dir
    dest_dir = args.dest_dir

    image_list = os.listdir(src_dir)
    image_list = [os.path.join(src_dir, id_) for id_ in image_list]
    sub_dataset = SubDataSet(image_list=image_list, image_dir=image_dir, label_dir=label_dir, annot_dir=annot_dir, dest_dir=dest_dir)
    sub_dataset.run()
