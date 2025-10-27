# TensorFlow Debugging Example - Common Issues and Fixes
import tensorflow as tf
import numpy as np

print("=== TensorFlow Debugging Examples ===")

# COMMON BUG 1: Dimension mismatch
print("\n1. Dimension Mismatch Fix:")
print("Original bug: model.add(Dense(64, input_shape=(784))) for MNIST data")
print("Fixed version:")

# Correct approach for MNIST
model_correct = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_correct.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
print("✅ Model compiled successfully - correct dimensions")

# COMMON BUG 2: Wrong loss function
print("\n2. Loss Function Fix:")
print("Bug: Using 'categorical_crossentropy' with integer labels")
print("Fix: Use 'sparse_categorical_crossentropy' for integer labels")

# Demonstrate the difference
y_true_integers = np.array([0, 1, 2, 1, 0])  # Integer labels
y_true_categorical = np.array([[1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0]])  # One-hot

print(f"Integer labels shape: {y_true_integers.shape}")
print(f"One-hot labels shape: {y_true_categorical.shape}")
print("✅ Use 'sparse_categorical_crossentropy' for integer labels")

# COMMON BUG 3: Data type issues
print("\n3. Data Type Fix:")
print("Bug: Training with integer images (0-255) without normalization")
print("Fix: Normalize to float32 (0-1)")

# Sample data
images_int = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
images_float = images_int.astype('float32') / 255.0

print(f"Original data type: {images_int.dtype}, range: [{images_int.min()}, {images_int.max()}]")
print(f"Fixed data type: {images_float.dtype}, range: [{images_float.min():.2f}, {images_float.max():.2f}]")
print("✅ Always normalize image data to 0-1 range")

print("\n=== ALL BUGS FIXED ===")
print("✅ Dimension mismatches resolved")
print("✅ Correct loss functions applied") 
print("✅ Data preprocessing implemented")
print("✅ Model ready for training")
