import dataloader.path
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(dataloader.path.MNIST, one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

in_units = 784
# 隐层节点数
h1_units = 300

# 权重初始化
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
y_ = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

# 定义结构
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 训练
tf.global_variables_initializer().run()

for i in range(5000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x: batch_x, y_: batch_y, keep_prob: 0.75})


# 评测
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
