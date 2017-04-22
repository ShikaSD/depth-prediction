from utils.nn import *


def model(input_tensor):
    with tf.variable_scope("model"):
        # Encoder (VGG based)
        conv1 = encoder_block(input_tensor, 32, kernel_size=(7, 7))
        print(conv1.get_shape().as_list())
        conv2 = encoder_block(conv1, 64, kernel_size=(5, 5))
        print(conv2.get_shape().as_list())
        conv3 = encoder_block(conv2, 128)
        print(conv3.get_shape().as_list())
        conv4 = encoder_block(conv3, 256)
        print(conv4.get_shape().as_list())
        conv5 = encoder_block(conv4, 512)
        print(conv5.get_shape().as_list())
        conv6 = encoder_block(conv5, 512)
        print(conv6.get_shape().as_list())
        conv7 = encoder_block(conv6, 512)
        print(conv7.get_shape().as_list())

        # Decoder
        deconv7 = decoder_block(conv7, 512, conv6, scale=2)
        print(deconv7.get_shape().as_list())
        deconv6 = decoder_block(deconv7, 512, conv5, scale=2)
        print(deconv6.get_shape().as_list())
        deconv5 = decoder_block(deconv6, 256, conv4, scale=2)
        print(deconv5.get_shape().as_list())
        deconv4, disp4 = decoder_block(deconv5, 128, conv3, scale=2, with_disp=True)
        print(deconv4.get_shape().as_list())
        deconv3, disp3 = decoder_block(deconv4, 64, tf.concat([conv2, disp4], 3), scale=2, with_disp=True)
        print(deconv3.get_shape().as_list())
        deconv2, disp2 = decoder_block(deconv3, 32, tf.concat([conv1, disp3], 3), scale=2, with_disp=True)
        print(deconv2.get_shape().as_list())
        deconv1, disp1 = decoder_block(deconv2, 16, disp2, scale=2, with_disp=True, upsample_disp=False)
        print(deconv1.get_shape().as_list())

        # Outputs
        outputs = [disp1, disp2, disp3, disp4]
        outputs_left  = [tf.expand_dims(x[:, :, :, 0], 3) for x in outputs]
        outputs_right = [tf.expand_dims(x[:, :, :, 1], 3) for x in outputs]

        return outputs, outputs_left, outputs_right


def loss(output_left, output_right, batch_left, batch_right):
    scaled_left  = scaled_batch(batch_left)
    scaled_right = scaled_batch(batch_right)

    # Generate images
    left  = [generate_image(scaled_right[i], -output_left[i]) for i in range(num_scales)]
    right = [generate_image(scaled_left[i],  output_right[i]) for i in range(num_scales)]

    # Calculate LR consistency
    left2right_disp = [generate_image(output_right[i], -output_left[i]) for i in range(num_scales)]
    right2left_disp = [generate_image(output_left[i],  output_right[i]) for i in range(num_scales)]

    # Calculate smoothness
    disparity_left_smooth = disp_smoothness(output_left, scaled_left)
    disparity_right_smooth = disp_smoothness(output_right, scaled_right)

    # Calculate loss
    # L1
    l1_left = [tf.reduce_mean(tf.abs(left[i] - scaled_left[i])) for i in range(num_scales)]
    l1_right = [tf.reduce_mean(tf.abs(right[i] - scaled_right[i])) for i in range(num_scales)]

    # SSIM
    ssim_left = [tf.reduce_mean(ssim(output_left[i], scaled_left[i])) for i in range(num_scales)]
    ssim_right = [tf.reduce_mean(ssim(output_right[i], scaled_right[i])) for i in range(num_scales)]

    # Weighted sum
    loss_left  = [0.85 * ssim_left[i] + 0.15 * l1_left[i] for i in range(num_scales)]
    loss_right = [0.85 * ssim_right[i] + 0.15 * l1_right[i] for i in range(num_scales)]
    image_loss = tf.add_n(loss_left + loss_right)

    # Disparity loss
    disp_left = [tf.reduce_mean(tf.abs(disparity_left_smooth[i])) / 2 ** i for i in range(num_scales)]
    disp_right = [tf.reduce_mean(tf.abs(disparity_right_smooth[i])) / 2 ** i for i in range(num_scales)]
    disp_loss = tf.add_n(disp_left + disp_right)

    # LR consitency loss
    lr_left = [tf.reduce_mean(tf.abs(right2left_disp[i] - output_left[i])) for i in range(4)]
    lr_right = [tf.reduce_mean(tf.abs(left2right_disp[i] - output_right[i])) for i in range(4)]
    lr_loss = tf.add_n(lr_left + lr_right)

    return image_loss + 0.1 * disp_loss + lr_loss


def summary(output_left, output_right, batch_left, batch_right, loss):
    collections = ['model']
    tf.summary.image("Left images", batch_left, max_outputs=4, collections=collections)
    tf.summary.image("Right images", batch_right, max_outputs=4, collections=collections)

    tf.summary.image('Disparity output left', output_left, max_outputs=4, collections=collections)
    tf.summary.image('Disparity output right', output_right, max_outputs=4, collections=collections)

    tf.summary.scalar("Loss", loss, collections=collections)