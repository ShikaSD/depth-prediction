import h5py
import numpy as np
import cv2

first_set = h5py.File("samples/nyu_depth_data_labeled.mat")
second_set = h5py.File("samples/nyu_depth_v2_labeled.mat")
depths = np.concatenate((np.asarray(first_set.get("depths")), np.asarray(second_set.get("depths"))))
images = np.concatenate((np.asarray(first_set.get("images")), np.asarray(second_set.get("images"))))

print(depths.shape)
print(images.shape)

depths = np.array([cv2.flip(np.rot90(x, k=3), 1) for x in depths])

images_reshaped = np.empty((images.shape[0], images.shape[3], images.shape[2], images.shape[1])).astype(np.uint8)
for x in range(0, images.shape[0]):
    image = images[x]
    for i in range(0, image.shape[0]):
        for j in range(0, image[i].shape[0]):
            for k in range(0, image[i, j].shape[0]):
                images_reshaped[x, k, j, i] = image[i, j, k]

images_reshaped.tofile(open("samples/images-0.np", "wb"))
depths.tofile(open("samples/depths-0.np", "wb"))
np.array([len(images_reshaped)]).tofile(open("samples/length-0.np", "wb"))