from collections import namedtuple

Label = namedtuple(
    'Label', ['name', 'id', 'classId', 'className', 'categoryId',
              'hasInstances', 'ignoreInEval', 'color'])

segnet_format_labels = {
    Label('Sky',                    0, 0,   'Sky',            0, True, False, (128, 128, 128)),
    Label('Building',               1, 1,   'Building',       1, True, False, (128, 0, 0)),
    Label('Pole',                   2, 2,   'Pole',           2, True, False, (192, 192, 128)),
    Label('Road',                   3, 3,   'Road',           3, True, False, (128, 64, 128)),
    Label('Pavement',               4, 4,   'Pavement',       4, True, False, (60, 40, 222)),
    Label('Tree',                   5, 5,   'Tree',           5, True, False, (128, 128, 0)),
    Label('SignSymbol',             6, 6,   'SignSymbol',     6, True, False, (192, 128, 128)),
    Label('Fence',                  7, 7,   'Fence',          7, True, False, (64, 64, 128)),
    Label('Car',                    8, 8,   'Car',            8, True, False, (64, 0, 128)),
    Label('Pedestrian',             9, 9,   'Pedestrian',     9, True, False, (64, 64, 0)),
    Label('Bicyclist',              10, 10, 'Bicyclist',      10, True, False, (0, 128, 192)),
    Label('Unlabelled',             11, 11, 'Unlabelled',     11, True, False, (0, 0, 0)),
}

segnet_lane_labels = {
    Label('Sky',                    0, 0,   'Sky',            0, True, False, (128, 128, 128)),
    Label('Building',               1, 1,   'Building',       1, True, False, (128, 0, 0)),
    Label('Pole',                   2, 2,   'Pole',           2, True, False, (192, 192, 128)),
    Label('Road',                   3, 3,   'Road',           3, True, False, (128, 64, 128)),
    Label('Pavement',               4, 4,   'Pavement',       4, True, False, (60, 40, 222)),
    Label('Tree',                   5, 5,   'Tree',           5, True, False, (128, 128, 0)),
    Label('SignSymbol',             6, 6,   'SignSymbol',     6, True, False, (192, 128, 128)),
    Label('Fence',                  7, 7,   'Fence',          7, True, False, (64, 64, 128)),
    Label('Car',                    8, 8,   'Car',            8, True, False, (64, 0, 128)),
    Label('Pedestrian',             9, 9,   'Pedestrian',     9, True, False, (64, 64, 0)),
    Label('Bicyclist',              10, 10, 'Bicyclist',      10, True, False, (0, 128, 192)),
    Label('Road_marking',           11, 11, 'Road_marking',   11, True, False, (255, 69, 0)),
    Label('Unlabelled',             12, 12, 'Unlabelled',     12, True, False, (0, 0, 0)),
}

segnet_labels = {
    Label('Animal',                 0, 0,   'Pedestrian',      9, True, False, (64, 128, 64)),
    Label('Archway',                1, 1,   'Building',        1, True, False, (192, 0, 128)),
    Label('Bicyclist',              2, 2,   'Bicyclist',       10,True, False, (0, 128, 192)),
    Label('Bridge',                 3, 3,   'Building',        1, True, False, (0, 128, 64)),
    Label('Building',               4, 4,   'Building',        1, True, False, (128, 0, 0)),
    Label('Car',                    5, 5,   'Car',             8, True, False, (64, 0, 128)),
    Label('CartLuggagePram',        6, 6,   'Car',             8, True, False, (64, 0, 192)),
    Label('Child',                  7, 7,   'Pedestrian',      9, True, False, (192, 128, 64)),
    Label('Column_Pole',            8, 8,   'Pole',            2, True, False, (192, 192, 128)),
    Label('Fence',                  9, 9,   'Fence',           7, True, False, (64, 64, 128)),
    Label('LaneMkgsDriv',           10, 10, 'Road_marking',    11,True, False, (128, 0, 192)),
    Label('LaneMkgsNonDriv',        11, 11, 'Road_marking',    11,True, False, (192, 0, 64)),
    Label('Misc_Text',              12, 12, 'SignSymbol',      6, True, False, (128, 128, 64)),
    Label('MotorcycleScooter',      13, 13, 'Car',             8, True, False, (192, 0, 192)),
    Label('OtherMoving',            14, 14, 'Pedestrian',      9, True, False, (128, 64, 64)),
    Label('ParkingBlock',           15, 15, 'Pavement',        4, True, False, (64, 192, 128)),
    Label('Pedestrian',             16, 16, 'Pedestrian',      9, True, False, (64, 64, 0)),
    Label('Road',                   17, 17, 'Road',            3, True, False, (128, 64, 128)),
    Label('RoadShoulder',           18, 18, 'Road',            3, True, False, (128, 128, 192)),
    Label('Sidewalk',               19, 19, 'Pavement',        4, True, False, (0, 0, 192)),
    Label('SignSymbol',             20, 20, 'SignSymbol',      6, True, False, (192, 128, 128)),
    Label('Sky',                    21, 21, 'Sky',             0, True, False, (128, 128, 128)),
    Label('SUVPickupTruck',         22, 22, 'Car',             8, True, False, (64, 128, 192)),
    Label('TrafficCone',            23, 23, 'SignSymbol',      6, True, False, (0, 0, 64)),
    Label('TrafficLight',           24, 24, 'SignSymbol',      6, True, False, (0, 64, 64)),
    Label('Train',                  25, 25, 'Car',             8, True, False, (192, 64, 128)),
    Label('Tree',                   26, 26, 'Tree',            5, True, False, (128, 128, 0)),
    Label('Truck_Bus',              27, 27, 'Car',             8, True, False, (192, 128, 192)),
    Label('Tunnel',                 28, 28, 'Road',            3, True, False, (64, 0, 64)),
    Label('VegetationMisc',         29, 29, 'Tree',            5, True, False, (192, 192, 0)),
    Label('Void',                   30, 30, 'Unlabelled',      12,True, False, (0, 0, 0)),
    Label('Wall',                   31, 31, 'Building',        1, True, False, (64, 192, 0)),
}


if __name__ == '__main__':
    label_txt = "../segnet_label.txt"
    label_id = 0
    with open(label_txt, "rb") as f:
        line = f.readline()
        while line:
            line = line.rstrip(b'\n')
            str_list = line.split(b"\t")
            color = str_list[0]
            name = str_list[len(str_list) - 1]
            color_list = color.split(b" ")
            r = int(color_list[0])
            g = int(color_list[1])
            b = int(color_list[2])
            print("Label('{}',           {}, {}, '{}',   {}, True, False, ({}, {}, {})),".
                  format(name.decode("UTF-8"), label_id, label_id, name.decode("UTF-8"), label_id, r, g, b))
            label_id += 1
            line = f.readline()
