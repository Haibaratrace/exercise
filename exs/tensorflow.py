import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
w = tf.Variable(tf.random_normal(shape=(784, 10), stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.zero(shape=[10], dtype=tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as ses:
    ses.run(init)
    print(ses.run(w))
