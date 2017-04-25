import tensorflow as tf
from tensorflow.contrib.layers import *
import numpy as np

from utils.bilinear import bilinear_sampler_1d_h


# def convolution2d(tensor, output_depth, kernel_size, activation_fn, stride=1):
#     padding = np.floor((kernel_size[0] - 1) / 2).astype(np.int32)
#     tensor = tf.pad(tensor, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
#     return conv2d(tensor, output_depth, kernel_size, stride=stride, activation_fn=activation_fn)


def _upsample(tensor, scale):
    shape = tensor.get_shape().as_list()
    h = shape[1]
    w = shape[2]
    return tf.image.resize_nearest_neighbor(tensor, size=[h * scale, w * scale])


def _disp(tensor):
    return .3 * convolution2d(tensor, 2, kernel_size=(3, 3), activation_fn=tf.nn.sigmoid)


def encoder_block(tensor, output_depth, kernel_size=(3, 3)):
    conv1 = convolution2d(tensor, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)
    conv2 = convolution2d(conv1,  output_depth, kernel_size=kernel_size, stride=2, activation_fn=tf.nn.elu)
    return conv2


def deconv_decoder_block(tensor, output_depth, concat_layer, kernel_size=(3, 3), scale=1, with_disp=False, upsample_disp=True):
    conv   = convolution2d_transpose(tensor, output_depth, kernel_size=kernel_size, stride=scale, activation_fn=tf.nn.elu)
    concat = tf.concat([conv, concat_layer], 3)
    fconv  = convolution2d(concat, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)

    if with_disp:
        conv_disp = _disp(fconv)
        return (fconv, conv_disp, _upsample(conv_disp, 2)) if upsample_disp else (fconv, conv_disp)

    return fconv


def upsample_decoder_block(tensor, output_depth, concat_layer, kernel_size=(3, 3), scale=1, with_disp=False, upsample_disp=True):
    upsample = _upsample(tensor, scale)
    conv     = convolution2d(upsample, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)
    concat   = tf.concat([conv, concat_layer], 3)
    fconv    = convolution2d(concat, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)

    if with_disp:
        conv_disp = _disp(fconv)
        return (fconv, conv_disp, _upsample(conv_disp, 2)) if upsample_disp else (fconv, conv_disp)

    return fconv

def generate_image(img, disp):
    return bilinear_sampler_1d_h(img, disp)

num_scales = 4


def scaled_batch(batch):
    scaled = [batch]
    shape = batch.get_shape().as_list()
    h = shape[1]
    w = shape[2]
    for i in range(num_scales - 1):
        scale = 2 ** (i + 1)
        scaled.append(tf.image.resize_area(batch, [tf.to_int32(h / scale), tf.to_int32(w / scale)]))
    return scaled


def _gradient_x(img):
    """
    Calculate smoothness by measuring a difference between left-right shifted images
    """
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    """
    Calculate smoothness by measuring a difference between left-right shifted images
    """
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def disp_smoothness(output, scaled):
    output_gradients_x = [_gradient_x(x) for x in output]
    output_gradients_y = [_gradient_y(x) for x in output]

    image_gradients_x = [_gradient_x(x) for x in scaled]
    image_gradients_y = [_gradient_y(x) for x in scaled]

    weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
    weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

    smoothness_x = [output_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [output_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y


def ssim(x, y):
    """
    Define SSIM with structure similar to https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py  
    Use avg pool instead of gaussian to speed up calculations
    Calculate structural dissimilarity as defined in https://en.wikipedia.org/wiki/Structural_similarity#Structural_Dissimilarity
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = avg_pool2d(x, (3, 3), stride=1, padding='VALID')
    mu_y = avg_pool2d(y, (3, 3), stride=1, padding='VALID')

    mu_xsq = tf.square(mu_x)
    mu_ysq = tf.square(mu_y)
    mu_xy = mu_x * mu_y
    sigma_x  = avg_pool2d(tf.square(x), (3, 3), stride=1, padding='VALID') - mu_xsq
    sigma_y  = avg_pool2d(tf.square(y), (3, 3), stride=1, padding='VALID') - mu_ysq
    sigma_xy = avg_pool2d(x * y, (3, 3), stride=1, padding='VALID') - mu_xy

    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (mu_xsq + mu_ysq + C1) * (sigma_x + sigma_y + C2)

    return tf.clip_by_value((1 - ssim) / 2, 0, 1)
