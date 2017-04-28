import tensorflow as tf
from datetime import datetime
import time
import math

num_batches = 100
batch_size = 32

def conv_op(input_op, name, kernel_height, kernel_width, kernel_outs,
            step_height, step_width, params):

    kernel_ins = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[kernel_height, kernel_width, kernel_ins, kernel_outs],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, step_height, step_width, 1),
                            padding='SAME')

        bias = tf.Variable(tf.constant(0.0, shape=[kernel_outs], dtype=tf.float32)
                           , trainable=True, name='b')

        activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)

        params += [kernel, bias]

        return activation


def fc_op(input_op, name, n_out, params):

    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        weight = tf.get_variable(scope+'w',
                                 shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        bias = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')

        activation = tf.nn.relu_layer(input_op, weight, bias, name=scope)

        params += [weight, bias]

        return activation

def mpool_op(input_op, name, kernel_height, kernel_width,
             step_height, step_width):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, step_height, step_width, 1],
                          padding='SAME',
                          name=name)

def inference_op(input_op, keep_prob):
    params = []

    conv1_1 = conv_op(input_op, name='conv1_1', kernel_height=3, kernel_width=3,
                      kernel_outs=64, step_height=1, step_width=1, params=params)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kernel_height=3, kernel_width=3,
                      kernel_outs=64, step_height=1, step_width=1, params=params)
    pool1 = mpool_op(conv1_2, name='pool1', kernel_height=2, kernel_width=2, step_height=2, step_width=2)


    conv2_1 = conv_op(pool1, name='conv2_1', kernel_height=3, kernel_width=3,
                      kernel_outs=128, step_height=1, step_width=1, params=params)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kernel_height=3, kernel_width=3,
                      kernel_outs=128, step_height=1, step_width=1, params=params)
    pool2 = mpool_op(conv2_2, name='pool2', kernel_height=2, kernel_width=2, step_height=2, step_width=2)


    conv3_1 = conv_op(pool2, name='conv3_1', kernel_height=3, kernel_width=3,
                      kernel_outs=256, step_height=1, step_width=1, params=params)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kernel_height=3, kernel_width=3,
                      kernel_outs=256, step_height=1, step_width=1, params=params)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kernel_height=3, kernel_width=3,
                      kernel_outs=256, step_height=1, step_width=1, params=params)
    pool3 = mpool_op(conv3_3, name='pool3', kernel_height=2, kernel_width=2, step_height=2, step_width=2)


    conv4_1 = conv_op(pool3, name='conv4_1', kernel_height=3, kernel_width=3,
                      kernel_outs=512, step_height=1, step_width=1, params=params)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kernel_height=3, kernel_width=3,
                      kernel_outs=512, step_height=1, step_width=1, params=params)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kernel_height=3, kernel_width=3,
                      kernel_outs=512, step_height=1, step_width=1, params=params)
    pool4 = mpool_op(conv4_3, name='pool4', kernel_height=2, kernel_width=2, step_height=2, step_width=2)


    conv5_1 = conv_op(pool4, name='conv5_1', kernel_height=3, kernel_width=3,
                      kernel_outs=512, step_height=1, step_width=1, params=params)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kernel_height=3, kernel_width=3,
                      kernel_outs=512, step_height=1, step_width=1, params=params)
    conv5_3 = conv_op(conv5_2, name='conv5_2', kernel_height=3, kernel_width=3,
                      kernel_outs=512, step_height=1, step_width=1, params=params)
    pool5 = mpool_op(conv5_3, name='pool5', kernel_height=2, kernel_width=2, step_height=2, step_width=2)

    shape = pool5.get_shape()
    n = shape[1].value * shape[2].value * shape[3].value
    flat = tf.reshape(pool5, [-1, n], name='flat')

    fc6 = fc_op(flat, name='fc6', n_out=4096, params=params)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, params=params)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, params=params)
    softmax = tf.nn.softmax(fc8)
    preditions = tf.argmax(softmax, 1)

    return preditions, softmax, fc8, params


def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size, image_size,
                                               3],
                                               dtype=tf.float32,
                                              stddev=1e-1))

        keep_prob = tf.placeholder(tf.float32)
        preditions, softmax, fc8, p = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, preditions, {keep_prob:1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob:1.0}, "Forward-backward")


if __name__ == '__main__':
    run_benchmark()