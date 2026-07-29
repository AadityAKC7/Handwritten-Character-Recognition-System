# study_dataset.py
import tensorflow as tf

# Load MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("=== MNIST Dataset Analysis ===")
print(f"Image shape: {x_train[0].shape}")
print(f"Image data type: {x_train.dtype}")
print(f"Pixel value range: {x_train.min()} to {x_train.max()}")
print(f"Number of training samples: {len(x_train)}")
print(f"Number of test samples: {len(x_test)}")
print(f"Number of unique classes: {len(set(y_train))}")
print(f"Class labels: {sorted(set(y_train))}")