from collections import namedtuple
import argparse
import os
import cv2

Label = namedtuple(
    'Label', ['name', 'id', 'classId', 'className', 'categoryId',
              'hasInstances', 'ignoreInEval', 'color'])

mapillary_instance_labels = [
    Label('unlabeled',              65, 12, 'Unlabelled',   0, False,   True,  (0, 0, 0)),
    Label('EgoVehicle',             64, 12, 'Unlabelled',   0, False,   False, (0, 0, 0)),
    Label('CarMount',               63, 11, 'Bicyclist',    0, False,   False, (32, 32, 32)),
    Label('WheeledSlow',            62, 11, 'Bicyclist',    1, True,    False, (0, 0, 192)),
    Label('Truck',                  61, 9,  'Car',          1, True,    False, (0, 0, 70)),
    Label('Trailer',                60, 9,  'Car',          1, True,    False, (0, 0, 110)),
    Label('OtherVehicle',           59, 9,  'Car',          1, True,    False, (128, 64, 64)),
    Label('OnRails',                58, 9,  'Car',          1, False,   False, (0, 80, 100)),
    Label('Motorcycle',             57, 9,  'Car',          1, True,    False, (0, 0, 230)),
    Label('Caravan',                56, 9,  'Car',          1, True,    False, (0, 0, 90)),
    Label('Car',                    55, 9,  'Car',          1, True,    False, (0, 0, 142)),
    Label('Bus',                    54, 9,  'Car',          1, True,    False, (0, 60, 100)),
    Label('Boat',                   53, 9,  'Car',          1, True,    False, (0, 0, 142)),
    Label('Bicycle',                52, 11, 'Bicyclist',    1, True,    False, (119, 11, 32)),
    Label('TrashCan',               51, 5,  'Pavement',     1, True,    False, (180, 165, 180)),
    Label('TrafficSign(Front)',     50, 7,  'SignSymbol',   1, True,    False, (220, 220, 0)),
    Label('TrafficSign(Back)',      49, 7,  'SignSymbol',   1, True,    False, (192, 192, 192)),
    Label('TrafficLight',           48, 7,  'SignSymbol',   1, True,    False, (250, 170, 30)),
    Label('UtilityPole',            47, 9,  'Car',          1, True,    False, (0, 0, 142)),
    Label('TrafficSignFrame',       46, 7,  'SignSymbol',   1, True,    False, (128, 128, 128)),
    Label('Pole',                   45, 2,  'Pole',         1, True,    False, (153, 153, 153)),
    Label('StreetLight',            44, 2,  'Pole',         1, True,    False, (210, 170, 100)),
    Label('Pothole',                43, 2,  'Pole',         1, False,   False, (170, 170, 170)),
    Label('PhoneBooth',             42, 9,  'Car',          1, True,    False, (0, 0, 142)),
    Label('Manhole',                41, 2,  'Pole',         1, True,    False, (170, 170, 170)),
    Label('Mailbox',                40, 7,  'SignSymbol',   1, True,    False, (33, 33, 33)),
    Label('JunctionBox',            39, 7,  'SignSymbol',   1, True,    False, (40, 40, 40)),
    Label('FireHydrant',            38, 7,  'SignSymbol',   1, True,    False, (100, 170, 30)),
    Label('CCTVCamera',             37, 7,  'SignSymbol',   1, True,    False, (222, 40, 40)),
    Label('CatchBasin',             36, 2,  'Pole',         1, True,    False, (170, 170, 170)),
    Label('Billboard',              35, 8,  'Fence',        1, True,    False, (220, 220, 220)),
    Label('BikeRack',               34, 12, 'Unlabelled',   1, True,    False, (0, 0, 0)),
    Label('Bench',                  33, 8,  'Fence',        1, True,    False, (250, 0, 30)),
    Label('Banner',                 32, 7,  'SignSymbol',   1, True,    False, (255, 255, 128)),
    Label('Water',                  31, 5,  'Pavement',     2, False,   False, (0, 170, 30)),
    Label('Vegetation',             30, 6,  'Tree',         2, False,   False, (107, 142, 35)),
    Label('Terrain',                29, 5,  'Pavement',     2, False,   False, (152, 251, 152)),
    Label('Snow',                   28, 5,  'Road_marking', 2, False,   False, (255, 255, 255)),
    Label('Sky',                    27, 0,  'Sky',          2, False,   False, (70, 130, 180)),
    Label('Sand',                   26, 9,  'Car',          2, False,   False, (128, 64, 64)),
    Label('Mountain',               25, 5,  'Pavement',     2, False,   False, (64, 170, 64)),
    Label('LaneMarking-General',    24, 3,  'Road_marking', 3, False,   False, (255, 255, 255)),
    Label('LaneMarking-Crosswalk',  23, 3,  'Road_marking', 3, True,    False, (200, 128, 128)),
    Label('OtherRider',             22, 11, 'Bicyclist',    4, True,    False, (255, 0, 0)),
    Label('Motorcyclist',           21, 11, 'Bicyclist',    4, True,    False, (255, 0, 0)),
    Label('Bicyclist',              20, 11, 'Bicyclist',    4, True,    False, (255, 0, 0)),
    Label('Person',                 19, 10, 'Pedestrian',   4, True,    False, (220, 20, 60)),
    Label('Tunnel',                 18, 1,  'Building',     5, False,   False, (150, 120, 90)),
    Label('Building',               17, 1,  'Building',     5, False,   False, (70, 70, 70)),
    Label('Bridge',                 16, 1,  'Building',     5, False,   False, (150, 100, 100)),
    Label('Sidewalk',               15, 3,  'Road_marking', 5, False,   False, (244, 35, 232)),
    Label('ServiceLane',            14, 4,  'Road',         5, False,   False, (110, 110, 110)),
    Label('Road',                   13, 4,  'Road',         5, False,   False, (128, 64, 128)),
    Label('RailTrack',              12, 4,  'Road',         5, False,   False, (230, 150, 140)),
    Label('PedestrianArea',         11, 4,  'Road',         5, False,   False, (96, 96, 96)),
    Label('Parking',                10, 3,  'Road_marking', 5, False,   False, (250, 170, 160)),
    Label('CurbCut',                9,  2,  'Pole',         5, False,   False, (170, 170, 170)),
    Label('Crosswalk-Plain',        8,  3,  'Road_marking', 5, True,    False, (140, 140, 200)),
    Label('BikeLane',               7,  3,  'Road_marking', 5, False,   False, (128, 64, 255)),
    Label('Wall',                   6,  1,  'Building',     5, False,   False, (102, 102, 156)),
    Label('Barrier',                5,  1,  'Building',     5, False,   False, (102, 102, 156)),
    Label('GuardRail',              4,  5,  'Pavement',     5, False,   False, (180, 165, 180)),
    Label('Fence',                  3,  8,  'Fence',        5, False,   False, (190, 153, 153)),
    Label('Curb',                   2,  5,  'Pavement',     5, False,   False, (196, 196, 196)),
    Label('GroundAnimal',           1,  10, 'Pedestrian',   6, True,    False, (0, 192, 0)),
    Label('Bird',                   0,  10, 'Pedestrian',   6, True,    False, (165, 42, 42)),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--check_dir', type=str, required=True)
    args = parser.parse_args()

    image_dir = ''
    if args.dir and args.dir != '' and os.path.exists(args.dir):
        image_dir = args.dir
        print(image_dir)

    check_dir = args.check_dir

    if image_dir:
        if not image_dir.endswith('/'):
            image_dir = image_dir + '/'
    # label_clr = {l.id: [l.name, l.color] for l in mapillary_instance_labels}
    #
    # for k, v in label_clr.items():
    #     if(k < 10):
    #         print("ID:" + str(k) + "    Name:" + v[0] + "    Color(" + str(v[1]) + ")")
    #     else:
    #         print("ID:" + str(k) + "   Name:" + v[0] + "    Color(" + str(v[1]) + ")")

    name_clr = {l.name: l.color for l in mapillary_instance_labels}
    check_class = [l.name for l in mapillary_instance_labels]
    check_clr = {name_clr[name]: name for name in check_class}
    find_clr = {name_clr[name]: 0 for name in check_class}
    check_result = {}

    check_result = {}
    image_list = os.listdir(image_dir)
    image_list.sort()

    check_file = os.path.join(check_dir, "check_file_all.txt")

    for _id in image_list:
        file_path = os.path.join(image_dir, _id)
        print(file_path)

        img = cv2.imread(filename=file_path)
        width = img.shape[1]
        height = img.shape[0]

        finish = True
        for x in range(width):
            for y in range(height):
                color = img[y, x]
                color = color[::-1]
                color = tuple(color)

                for k, v in find_clr.items():
                    if not v:
                        finish = False
                        break

                if finish:
                    break

                if color in check_clr:
                    name = check_clr[color]
                    if find_clr[color] >= 10:
                        continue
                    find_clr[color] += 1
                    key = name + "_" + str(find_clr[color])
                    msg = "name:{}, file:{}, x,y:[{},{}],color:[{}]\n".format(name, file_path, x, y, str(color))
                    check_result[key] = msg

            if finish:
                break

        items = check_result.items()
        items.sort()
        for key, value in items:
            output = "{}/{}\n".format(key, value)
            print(output)

        if finish:
            break

    with open(check_file, "wb") as f:
        items = check_result.items()
        items.sort()

        for key, value in items:
            output = "{}/{}\n".format(key, value)
            f.write(output)
