import shutil
import os

kitti_dataset_folders = list(map(lambda x: "kitti/" + x + "/", list(filter(lambda x: x.endswith("sync"), os.listdir("kitti")))))
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
