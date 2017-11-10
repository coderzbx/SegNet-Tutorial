from SelfScripts.segnet_label import segnet_labels
from PIL import Image
import cv2

if __name__ == '__main__':
    clr_label = {l.color: l.categoryId for l in segnet_labels}
    image_path = '/SegNet/CamVid_lane/trainlabel/0001TP_006690_L.png'
    result_path = '/SegNet/CamVid_lane/test.png'

    print(image_path)

    img = cv2.imread(image_path)

    width = img.shape[1]
    height = img.shape[0]

    anna_img = Image.new('L', (width, height))

    img_data = anna_img.load()
    # for x in range(width):
    #     for y in range(height):
    #         color = img[y, x]
    #         color = color[::-1]
    #         color = tuple(color)
    #         if color in clr_label:
    #             label_id = clr_label[color]
    #             img_data[x, y] = label_id
    #         else:
    #             print(color)
    #             img_data[x, y] = 13
    #
    # anna_img.save(result_path)

    # test_image = Image.open(result_path)
    image_path = '/SegNet/CamVid_lane/trainannot/0001TP_006690_L.png'
    test_image1 = Image.open(image_path)

    image_path = '/opt/segnet_server/models/label_clr.png'
    test_image2 = Image.open(image_path)

    image_path = '/SegNet/CamVid_lane/label_clr.png'
    test_image3 = Image.open(image_path)
    print('Finish')