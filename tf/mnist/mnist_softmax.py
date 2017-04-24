# 导入mnist数据
import dataloader.path
from tensorflow.examples.tutorials.mnist import input_data

print(dataloader.path.MNIST)
mnist = input_data.read_data_sets(dataloader.path.MNIST, one_hot=True)



import tensorflow as tf

sess = tf.InteractiveSession()

# 数据输入设置
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 参数初始化设置
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 定义前向计算
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),
                                              reduction_indices=[1]))

# 定义优化方法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 真正的参数初始化
tf.global_variables_initializer().run()

# 随机梯度下降
for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch_x, y_: batch_y})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 输入label与评测流程 计算结果
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

