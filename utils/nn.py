import tensorflow as tf
from tensorflow.contrib.layers import *

from utils.bilinear import bilinear_sampler_1d_h


def _upsample(tensor, scale):
    shape = tf.shape(tensor)
    h = shape[1]
    w = shape[2]
    return tf.image.resize_nearest_neighbor(tensor, [h * scale, w * scale])


def _disp(tensor):
    return 0.3 * convolution2d(tensor, 2, kernel_size=(3, 3), activation_fn=tf.nn.sigmoid)


def encoder_block(tensor, output_depth, kernel_size=(3, 3)):
    conv1 = convolution2d(tensor, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)
    conv2 = convolution2d(conv1,  output_depth, kernel_size=kernel_size, stride=2, activation_fn=tf.nn.elu)
    return conv2


def decoder_block(tensor, output_depth, concat_layer, kernel_size=(3, 3), scale=1, with_disp=False, upsample_disp=True):
    conv   = convolution2d_transpose(tensor, output_depth, kernel_size=kernel_size, stride=scale, activation_fn=tf.nn.elu)
    concat = tf.concat([conv, concat_layer], 3)
    fconv  = convolution2d(concat, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)

    if with_disp:
        return fconv, _upsample(_disp(tensor), 2) if upsample_disp else _disp(tensor)

    return fconv


def generate_image(img, disp):
    return bilinear_sampler_1d_h(img, disp)

num_scales = 4


def scaled_batch(batch):
    scaled = [batch]
    shape = tf.shape(batch)
    h = shape[1]
    w = shape[2]
    for i in range(num_scales - 1):
        scale = (i + 1) ** 2
        scaled.append(tf.image.resize_area(batch, [h / scale, w / scale]))
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
