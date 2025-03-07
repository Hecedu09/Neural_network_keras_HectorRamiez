import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input  # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore

def load_data():
    """
    Loads the MNIST dataset and returns the training and test sets.
    """
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
    return train_data_x, train_labels_y, test_data_x, test_labels_y

def preprocess_data(data_x, labels_y):
    """
    Preprocesses the data:
    - Flattens 28x28 images into a 784-element vector.
    - Normalizes pixel values to a range of 0 to 1.
    - Converts labels to one-hot encoding.
    """
    x = data_x.reshape(data_x.shape[0], 28 * 28).astype('float32') / 255
    y = to_categorical(labels_y)
    return x, y

def visualize_example(data_x, index=1):
    """
    Displays an example image from the dataset with its label.
    """
    plt.imshow(data_x[index], cmap='gray')
    plt.title(f"Example Image - Index: {index}")
    plt.show()

def build_model():
    """
    Builds and compiles the neural network model for digit classification.
    """
    model = Sequential([
        Input(shape=(28 * 28,)),  # Input layer with 784 nodes (flattened 28x28 pixels)
        Dense(512, activation='relu'),  # Hidden layer with 512 neurons and ReLU activation
        Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit)
    ])
    
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',  # Loss function for multi-class classification
        metrics=['accuracy']  # Evaluation based on accuracy
    )
    return model

def train_and_evaluate():
    """
    Trains the neural network model and evaluates it on test data.
    """
    # Load and visualize data
    train_data_x, train_labels_y, test_data_x, test_labels_y = load_data()
    print("Training data shape:", train_data_x.shape)
    print("Label of the first example:", train_labels_y[1])
    visualize_example(train_data_x)
    
    # Data preprocessing
    x_train, y_train = preprocess_data(train_data_x, train_labels_y)
    x_test, y_test = preprocess_data(test_data_x, test_labels_y)
    
    # Model construction
    model = build_model()
    print("Model Summary:")
    model.summary()
    
    # Training
    print("Training the neural network...")
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    
    # Model evaluation
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

plt.show()
