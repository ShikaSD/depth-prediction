import cv2
import numpy as np
import gc
from scipy.sparse import *
from scipy.sparse.linalg import *


def fix_depth_map(bgr, depth):
    alpha = 1

    known_values = depth[(depth != 0) & (depth != 10)]

    max_depth = np.amax(known_values)
    depth = np.divide(depth, float(max_depth))
    depth[depth > 1] = 1

    h, w = depth.shape
    num_pix = h * w
    inds_m = np.reshape(range(0, num_pix), (h, w))
    gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    win_rad = 1
    length = 0
    abs_img_index = 0
    cols = np.zeros((num_pix * (2 * win_rad + 1) ** 2))
    rows = np.zeros_like(cols)
    vals = np.zeros_like(cols)
    gvals = np.zeros_like(cols)

    for j in range(0, w):
        for i in range(0, h):
            points_in_win = 0
            for ii in range(max(0, i - win_rad), min(i + win_rad, h)):
                for jj in range(max(0, j - win_rad), min(j + win_rad, w)):
                    if ii == i and jj == j:
                        continue

                    length += 1
                    points_in_win += 1
                    rows[length] = abs_img_index
                    cols[length] = inds_m[ii, jj]
                    gvals[points_in_win] = gray_img[ii, jj]

            current_value = gray_img[i, j]
            gvals[points_in_win] = current_value
            c_var = np.mean((gvals[points_in_win + 1] - np.mean(gvals[points_in_win + 1])) ** 2)
            c_sig = c_var * 0.6
            mgv = np.min((gvals[points_in_win + 1] - current_value) ** 2)
            if c_sig < (-mgv / np.log(0.01)):
                c_sig = -mgv / np.log(0.01)

            if c_sig < 0.000002:
                c_sig = 0.000002

            gvals[0: points_in_win + 1] = np.exp(-(gvals[0: points_in_win + 1] - current_value) ** 2 / c_sig)
            gvals[0: points_in_win + 1] /= np.sum(gvals[0: points_in_win + 1])
            vals[length - points_in_win - 1: length] = -gvals[0: points_in_win + 1]

            length += 1
            rows[length - 1] = abs_img_index
            cols[length - 1] = abs_img_index
            vals[length - 1] = 1

            abs_img_index += 1

    vals = vals[0:length - 1]
    cols = cols[0:length - 1]
    rows = rows[0:length - 1]
    a = coo_matrix((vals, (rows, cols)), shape=(num_pix, num_pix))

    vals = known_values[:]
    cols = range(0, np.size(known_values))
    rows = range(0, np.size(known_values))
    g = coo_matrix((vals, (rows, cols)), shape=(num_pix, num_pix))

    right_part = vals * np.reshape(depth, (np.size(depth),))
    left_part = a + g
    gc.collect()
    new_vals = spsolve(left_part, right_part)
    new_vals = np.reshape(new_vals, (h, w))

    print(new_vals, max_depth)

    return (new_vals * max_depth).astype(int)
