import h5py
import numpy as np
import pickle

data = h5py.File("samples/nyu_depth_v2_labeled.mat")

depths = np.asarray(data.get("depths"))
images = np.asarray(data.get("images"))

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

pickle.dump({"depths": depths, "images": images_reshaped}, open("samples/dataset.p", "wb"))