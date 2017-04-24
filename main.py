from model import *
import time


def preprocess_test_images(left_image, right_image):
    with tf.variable_scope("preprocess", reuse=True):
        # flip images
        should_flip = tf.random_uniform([], 0, 1)
        left_image, right_image = tf.cond(
            tf.greater(should_flip, 0.5),
            lambda: [tf.image.flip_left_right(right_image), tf.image.flip_left_right(left_image)],
            lambda: [left_image, right_image])

        # augment images
        should_augment = tf.Variable(tf.random_uniform([], 0, 1), trainable=False)
        left_image, right_image = tf.cond(tf.greater(should_augment, 0.5), lambda: augment(left_image, right_image),
                                          lambda: (left_image, right_image))

        left_image.set_shape([None, None, 3])
        right_image.set_shape([None, None, 3])

        return left_image, right_image


def augment(left_image, right_image):
    with tf.variable_scope("augment", reuse=True):
        # shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        shape = tf.shape(left_image)
        white = tf.ones([shape[0], shape[1]])
        color_mask = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_mask
        right_image_aug *= color_mask

        # normalize
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug


def read_images(files, input_width=512, input_height=512, batch_size=32):
    with tf.variable_scope("load_images", reuse=True):
        input_queue = tf.train.string_input_producer(files, shuffle=False)
        reader = tf.TextLineReader()
        _, path = reader.read(input_queue)

        splits = tf.string_split([path], ";").values
        # splits = tf.Print(splits, [splits], message="Batch values")
        left_image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(splits[0])), tf.float32)
        right_image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(splits[1])), tf.float32)

        left_image = tf.image.resize_images(left_image, [input_height, input_width], tf.image.ResizeMethod.AREA)
        right_image = tf.image.resize_images(right_image, [input_height, input_width], tf.image.ResizeMethod.AREA)

        return tf.train.shuffle_batch(
            preprocess_test_images(left_image, right_image),
            batch_size,
            128 + 4 * batch_size,
            128,
            4,
            allow_smaller_final_batch=True)


def count_lines(filenames):
    counter = 0
    for filename in filenames:
        with open(filename) as file:
            for lines in file:
                counter += 1

    return counter


def train(run_num):
    logdir = "logs/run%d" % run_num

    train_filenames = ["kitti/train.txt"]
    train_length = count_lines(train_filenames)
    test_filenames = ["kitti/test.txt"]

    rate = 1e-4

    EPOCHS = 50
    BATCH_SIZE = 32

    global_step = tf.Variable(0, trainable=False)

    boundaries = [np.int32((3 / 5.0) * EPOCHS), np.int32((4 / 5.0) * EPOCHS)]
    values = [rate, rate / 2, rate / 4]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batch_x, batch_y = read_images(train_filenames, batch_size=BATCH_SIZE)
    outputs, outputs_left, outputs_right = model(batch_x)
    loss_op = loss(outputs_left, outputs_right, batch_x, batch_y)

    grads = optimizer.compute_gradients(loss_op)
    grads_apply_op = optimizer.apply_gradients(grads, global_step=global_step)

    tf.summary.scalar('learning_rate', rate, ['model'])
    tf.summary.scalar('loss', loss_op, ['model'])

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, EPOCHS):
            before_op_time = time.time()
            _, loss_value = sess.run([grads_apply_op, loss_op])
            duration = time.time() - before_op_time
            if step:
                examples_per_sec = train_length / duration
                time_so_far = (time.time() - start_time)
                training_time_left = (EPOCHS / step - 1.0) * time_so_far
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}s | time left: {:.2f}s'
                print(print_string.format(step, examples_per_sec, loss_value, time_so_far, training_time_left))

                summary(outputs_left, outputs_right, batch_x, batch_y, loss_op, step)
                summary_op = tf.summary.merge_all('model')
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

                if step % 5 == 0:
                    saver.save(sess, logdir + '/model.cpkt', global_step=step)

        saver.save(sess, 'logs/model.cpkt', global_step=EPOCHS)

        coordinator.request_stop()
        coordinator.join(threads)

train(1)
