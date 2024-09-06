# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Step 1: Load and preprocess the MNIST dataset
# MNIST dataset contains handwritten digits (0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images (scale pixel values to range [0, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the data to match input format (28x28 images, flattened into 784)
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 2: Build a simple neural network model
model = models.Sequential()

# Input layer with 784 neurons (for the 28x28 image size)
model.add(layers.Input(shape=(28 * 28,)))

# Hidden layer with 128 neurons and ReLU activation
model.add(layers.Dense(128, activation='relu'))

# Output layer with 10 neurons (one for each digit) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Step 3: Compile the model
# We use 'categorical_crossentropy' as the loss function for multi-class classification
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 4: Train the model
# Train the model on the training data
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Step 5: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
