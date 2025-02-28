import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['DEVICE_NAME'] = 'cpu'  # Force CPU usage

import tensorflow as tf

# Disable GPU devices
tf.config.set_visible_devices([], 'GPU')

# Print TensorFlow information
print(f"TensorFlow version: {tf.__version__}")
print("Available devices:", tf.config.list_physical_devices())

# Try a simple computation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)
print("\nSimple matrix multiplication test:")
print(c.numpy()) 