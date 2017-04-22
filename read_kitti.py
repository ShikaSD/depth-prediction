import shutil
import os

kitti_dataset_folders = [
    'kitti/2011_09_26_drive_0001_sync/',
    'kitti/2011_09_26_drive_0009_sync/',
    'kitti/2011_09_26_drive_0084_sync/',
    'kitti/2011_09_26_drive_0093_sync/',
    'kitti/2011_09_26_drive_0096_sync/',
    'kitti/2011_09_26_drive_0117_sync/',
    'kitti/2011_09_28_drive_0002_sync/',
    'kitti/2011_09_29_drive_0071_sync/']
output = []

for folder in kitti_dataset_folders:
    shutil.rmtree(folder + 'image_00', ignore_errors=True)
    shutil.rmtree(folder + 'image_01', ignore_errors=True)
    shutil.rmtree(folder + 'oxts', ignore_errors=True)
    shutil.rmtree(folder + 'velodyne_points', ignore_errors=True)

    left = list(filter(lambda x: x.endswith('.jpg'), list(map(lambda x: folder + 'image_02/data/' + x, os.listdir(folder + 'image_02/data/')))))
    right = list(filter(lambda x: x.endswith('.jpg'), list(map(lambda x: folder + 'image_03/data/' + x, os.listdir(folder + 'image_03/data/')))))

    output += list(map(lambda x: x[0] + ';' + x[1], list(zip(left, right))))

print("Found %d pairs" % len(output))
test_split = int(0.2 * len(output))
with open("kitti/test.txt", "w") as f:
    f.write("\n".join(output[:test_split]))

with open("kitti/train.txt", "w") as f:
    f.write("\n".join(output[test_split:]))
