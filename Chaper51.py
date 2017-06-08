from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\\tmp\\tensorflow\\mnist\\input_data", one_hot=True)

print("Training data size: ", mnist.train.num_examples)
print("Validation data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)
print("Example training data: ", mnist.train.images[0])
print("Example training data label: ", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print("X shape:", xs.shape)
print("Y shape:", ys.shape)
