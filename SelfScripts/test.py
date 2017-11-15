from SelfScripts.segnet_label import segnet_labels
from PIL import Image
import cv2

if __name__ == '__main__':
    id_name_dict = {1: "tesafdgs", 2: "ferfg", 10: "sdsdets"}
    items = {1: [4333, 39240, 32], 10: [452, 564343, 5]}
    for k, (v1, v2, v3) in items.items():
        name = id_name_dict[k]
        print('class name:%20s  id:%3s  fileCount:%5s  pixel:%5s  /%10s' % (name, str(k), str(v3), str(v1), str(v2)))
        # print("class name:{} id:{} fileCount:{} pixel:{}/{}".format(name, k, v3, v1, v2))

    clr_label = {l.color: l.categoryId for l in segnet_labels}
    image_path = '/SegNet/CamVid_lane/trainlabel/0001TP_006690_L.png'
    result_path = '/SegNet/CamVid_lane/test.png'

    file_id = '0001TP_006690_L.png'
    file_id_list = str(file_id).split(".")
    file_id = file_id_list[0]
    file_id = file_id + ".jpg"

    print(image_path)

    img = cv2.imread(image_path)

    width = img.shape[1]
    height = img.shape[0]

    x1 = width/2 - 5
    x2 = width/2 + 5
    y1 = y2 = height/2

    x3 = x4 = width/2
    y3 = height/2 - 5
    y4 = height/2 + 5
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 1)

    cv2.imwrite(result_path, img)

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