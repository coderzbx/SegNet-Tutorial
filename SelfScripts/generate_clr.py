from PIL import Image
from SelfScripts.segnet_label import segnet_lane_labels

if __name__ == '__main__':
    label_clr = {l.id: l.color for l in segnet_lane_labels}

    clr_image = '/SegNet/CamVid_lane/label_clr.png'
    width = 256
    height = 1
    image = Image.new("RGBA", (width, height), (0, 0, 0))
    image_data = image.load()

    class_count = len(label_clr)
    for i in range(class_count):
        image_data[i, 0] = label_clr[i]

    image.save(clr_image)
