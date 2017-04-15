import os
import cv2
from fix_depth_maps import fix_depth_map
import numpy as np


def get_full_path(folder, image_list):
    return list(map(lambda x: folder + "/" + x, image_list))


def get_time(filename):
    return int(filename.split('-')[2].split('.')[0])

dataset_folders = ['samples/basements/basement_0001a']

for folder in dataset_folders:
    files = os.listdir(folder)
    images = sorted(list(filter(lambda x: x.endswith('.ppm'), files)))
    depths = sorted(list(filter(lambda x: x.endswith('.pgm'), files)))

    pairs = {}

    start = 0
    for image in images:
        t1 = get_time(image)
        min_t2 = float("inf")
        closest_pair = ""
        for depth in depths:
            t2 = abs(get_time(depth) - t1)
            if min_t2 > t2:
                min_t2 = t2
                closest_pair = depth

        pairs[closest_pair] = image

    values = list(pairs.values())
    keys = list(pairs.keys())
    images = list(filter(lambda x: x not in values, images))
    depths = list(filter(lambda x: x not in keys, depths))

    for image in get_full_path(folder, images):
        os.remove(image)

    for depth in get_full_path(folder, depths):
        os.remove(depth)

    files = os.listdir(folder)
    for file in files:
        if file not in keys and file not in values:
            os.remove(get_full_path(folder, [file])[0])

    dp_path = get_full_path(folder, [keys[0]])[0]
    img_path = get_full_path(folder, [values[0]])[0]
    img = cv2.resize(cv2.imread(img_path, -1), (70, 50))
    dp = cv2.resize(cv2.imread(dp_path, -1), (70, 50))
    cv2.imshow("depth1", dp)
    cv2.imshow("rgb1", img)
    print(np.amin(cv2.imread(dp_path, -1)), np.amax(cv2.imread(dp_path, -1)))
    cv2.imshow("depth2", fix_depth_map(img, dp))
    cv2.waitKey(0)
    cv2.destroyAllWindows()