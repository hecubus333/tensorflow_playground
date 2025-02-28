import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create a simple neural network model
model = tf.keras.Sequential([
    # Flatten the 28x28 images into a 784-dimensional vector
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Add a dense hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Add dropout to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    
    # Output layer with 10 neurons (one for each digit) and softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# Evaluate the model
print("\nEvaluating the model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Function to display an image and its prediction
def display_prediction(image, true_label, predictions):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    predicted_label = np.argmax(predictions)
    confidence = predictions[predicted_label]
    title = f'Prediction: {predicted_label}\nTrue Label: {true_label}\nConfidence: {confidence:.2f}'
    plt.title(title)
    plt.show()

# Make some predictions on test data
print("\nMaking predictions on test images...")
num_examples = 5
test_indices = np.random.randint(0, len(x_test), num_examples)

for idx in test_indices:
    # Get the image and make a prediction
    image = x_test[idx]
    true_label = y_test[idx]
    predictions = model.predict(image.reshape(1, 28, 28), verbose=0)[0]
    
    # Display the image and prediction
    display_prediction(image, true_label, predictions) 