import tensorflow as tf

# Basic addition
result = tf.add(1, 2).numpy()
print(result)

# Create and print a string constant
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy())