from utils.nn import *


def model(input_tensor):
    with tf.variable_scope("model"):
        # Encoder (VGG based)
        conv1 = encoder_block(input_tensor, 32, kernel_size=(7, 7))
        conv2 = encoder_block(conv1, 64, kernel_size=(5, 5))
        conv3 = encoder_block(conv2, 128)
        conv4 = encoder_block(conv3, 256)
        conv5 = encoder_block(conv4, 512)
        conv6 = encoder_block(conv5, 512)
        conv7 = encoder_block(conv6, 512)

        # Decoder
        deconv7 = decoder_block(conv7, 512, conv6, scale=2)
        deconv6 = decoder_block(deconv7, 512, conv5, scale=2)
        deconv5 = decoder_block(deconv6, 256, conv4, scale=2)
        deconv4, disp4 = decoder_block(deconv5, 128, conv3, scale=2, with_disp=True)
        deconv3, disp3 = decoder_block(deconv4, 64, tf.concat([conv2, disp4], 3), scale=2, with_disp=True)
        deconv2, disp2 = decoder_block(deconv3, 32, tf.concat([conv1, disp3], 3), scale=2, with_disp=True)
        deconv1, disp1 = decoder_block(deconv2, 16, disp2, scale=2, with_disp=True, upsample_disp=False)

        # Outputs
        outputs = [disp1, disp2, disp3, disp4]
        outputs_left  = [tf.expand_dims(x[:, :, :, 0], 3) for x in outputs]
        outputs_right = [tf.expand_dims(x[:, :, :, 1], 3) for x in outputs]

        return outputs, outputs_left, outputs_right


def loss(output, output_left, output_right, batch_left, batch_right):
    scaled_left  = scaled_batch(batch_left)
    scaled_right = scaled_batch(batch_right)

    # Generate images
    left  = [generate_image(scaled_right[i], -output_left[i]) for i in range(num_scales)]
    right = [generate_image(scaled_left[i], output_right[i]) for i in range(num_scales)]

    # Calculate LR consistency
    left_right_disparity = [generate_image(output_right[i], -output_left[i]) for i in range(num_scales)]
    right_left_disparity = [generate_image(output_left[i], output_right[i]) for i in range(num_scales)]

    # Calculate smoothness
    disparity_left_smooth = disp_smoothness(output_left, batch_left)
    disparity_right_smooth = disp_smoothness(output_right, batch_right)

    # Calculate loss
    # L1
    l1_left = [tf.reduce_mean(tf.abs(left[i] - scaled_left[i])) for i in range(4)]
    l1_right = [tf.reduce_mean(tf.abs(right[i] - scaled_right[i])) for i in range(4)]

    # SSIM

