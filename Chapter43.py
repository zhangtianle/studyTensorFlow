# page 90

import tensorflow as tf

# 定义变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)

# step迭代轮数
step = tf.Variable(0, trainable=False)

# 定义滑动平均类
ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))

    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
