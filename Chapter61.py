# page 146 卷积层样例

import tensorflow as tf

# 创建过滤器的权重和偏置，四维矩阵，前两维过滤器的尺寸，第三维当前层的深度，第四维过滤器的深度
filter_weight = tf.get_variable('weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

biases = tf.get_variable('biases', [16], initializer=tf.truncated_normal_initializer(0.1))

conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

bias = tf.nn.bias_add(conv, biases)

# 将计算结果通过ReLu激活函数完成去线性化
actived_conv = tf.nn.relu(bias)
