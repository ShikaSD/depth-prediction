import tensorflow as tf
from tensorflow.contrib.layers import *
from bilinear import bilinear_sampler_1d_h


def _upsample(tensor, scale):
    shape = tf.shape(tensor)
    h = shape[1]
    w = shape[2]
    return tf.image.resize_nearest_neighbor(tensor, [h * scale, w * scale])


def _disp(tensor):
    return 0.3 * convolution2d(tensor, 2, kernel_size=(3, 3), activation_fn=tf.nn.sigmoid)


def _encoder_block(tensor, output_depth, kernel_size=(3, 3)):
    conv1 = convolution2d(tensor, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)
    conv2 = convolution2d(conv1,  output_depth, kernel_size=kernel_size, stride=2, activation_fn=tf.nn.elu)
    return conv2


def _decoder_block(tensor, output_depth, concat_layer, kernel_size=(3, 3), scale=1, with_disp=False, upsample_disp=True):
    conv   = convolution2d_transpose(tensor, output_depth, kernel_size=kernel_size, stride=scale, activation_fn=tf.nn.elu)
    concat = tf.concat([conv, concat_layer], 3)
    fconv  = convolution2d(concat, output_depth, kernel_size=kernel_size, activation_fn=tf.nn.elu)

    if with_disp:
        return fconv, _upsample(_disp(tensor), 2) if upsample_disp else _disp(tensor)

    return fconv


def _generate_image(img, disp):
    return bilinear_sampler_1d_h(img, disp)


def model(input_tensor):

    with tf.variable_scope("model"):
        # Encoder (VGG based)
        conv1 = _encoder_block(input_tensor, 32, kernel_size=(7, 7))
        conv2 = _encoder_block(conv1, 64, kernel_size=(5, 5))
        conv3 = _encoder_block(conv2, 128)
        conv4 = _encoder_block(conv3, 256)
        conv5 = _encoder_block(conv4, 512)
        conv6 = _encoder_block(conv5, 512)
        conv7 = _encoder_block(conv6, 512)

        # Decoder
        deconv7 = _decoder_block(conv7,   512, conv6, scale=2)
        deconv6 = _decoder_block(deconv7, 512, conv5, scale=2)
        deconv5 = _decoder_block(deconv6, 256, conv4, scale=2)
        deconv4, disp4 = _decoder_block(deconv5, 128, conv3, scale=2, with_disp=True)
        deconv3, disp3 = _decoder_block(deconv4, 64,  tf.concat([conv2, disp4], 3), scale=2, with_disp=True)
        deconv2, disp2 = _decoder_block(deconv3, 32,  tf.concat([conv1, disp3], 3), scale=2, with_disp=True)
        deconv1, disp1 = _decoder_block(deconv2, 16,  disp2, scale=2, with_disp=True, upsample_disp=False)
