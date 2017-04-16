from cv2.ximgproc import jointBilateralFilter
import cv2
import numpy as np


def fix_depth_map(bgr, depth):
    space_sigma = 0.1
    range_sigma = 0.001

    max_depth_obs = max(depth[(depth != 0) & (depth != 10)])

    img_depth = depth / max_depth_obs
    img_depth[img_depth > 1] = 1
    img_depth = (img_depth * 255).astype(np.uint8)

    filtered = jointBilateralFilter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), img_depth, -1, range_sigma, space_sigma)

    return ((filtered) * max_depth_obs).astype(np.uint8)
