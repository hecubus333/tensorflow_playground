import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Function to create model
def create_model(use_dropout=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        # Only add dropout in one model
        tf.keras.layers.Dropout(0.2) if use_dropout else tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create two models
model_with_dropout = create_model(use_dropout=True)
model_without_dropout = create_model(use_dropout=False)

# Train both models
print("\nTraining model WITH dropout...")
history_with_dropout = model_with_dropout.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

print("\nTraining model WITHOUT dropout...")
history_without_dropout = model_without_dropout.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history_with_dropout.history['accuracy'], label='With Dropout')
plt.plot(history_without_dropout.history['accuracy'], label='Without Dropout')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_with_dropout.history['val_accuracy'], label='With Dropout')
plt.plot(history_without_dropout.history['val_accuracy'], label='Without Dropout')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Print final accuracies
print("\nFinal Results:")
_, test_acc_with = model_with_dropout.evaluate(x_test, y_test, verbose=0)
_, test_acc_without = model_without_dropout.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy with dropout: {test_acc_with:.4f}")
print(f"Test accuracy without dropout: {test_acc_without:.4f}") 