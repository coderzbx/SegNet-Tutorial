from PIL import Image
from SelfScripts.segnet_label import segnet_format_labels

if __name__ == '__main__':
    label_clr = {l.id: l.color for l in segnet_format_labels}

    clr_image = '/SegNet/CamVid_only_lane/label_clr_simple.png'
    width = 256
    height = 1
    image = Image.new("RGBA", (width, height), (0, 0, 0))
    image_data = image.load()

    class_count = len(label_clr)
    image_data[0, 0] = (255, 255, 255)
    image_data[1, 0] = label_clr[11]
    image_data[2, 0] = (0, 0, 0)

    image.save(clr_image)
