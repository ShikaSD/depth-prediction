from tensorflow.contrib.layers import *
import tensorflow as tf


def SSIM(self, x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)