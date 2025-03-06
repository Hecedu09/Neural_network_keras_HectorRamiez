import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input  # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore

def cargar_datos():
    """
    Carga el dataset MNIST y devuelve los conjuntos de entrenamiento y prueba.
    """
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
    return train_data_x, train_labels_y, test_data_x, test_labels_y

def preprocesar_datos(data_x, labels_y):
    """
    Preprocesa los datos:
    - Aplana las imágenes de 28x28 a un vector de 784 elementos.
    - Normaliza los valores de los píxeles a un rango de 0 a 1.
    - Convierte las etiquetas en codificación one-hot.
    """
    x = data_x.reshape(data_x.shape[0], 28 * 28).astype('float32') / 255
    y = to_categorical(labels_y)
    return x, y

def visualizar_ejemplo(data_x, index=1):
    """
    Muestra una imagen de ejemplo del dataset con su etiqueta.
    """
    plt.imshow(data_x[index], cmap='gray')
    plt.title(f"Ejemplo de imagen - Índice: {index}")
    plt.show()

def construir_modelo():
    """
    Construye y compila el modelo de red neuronal para clasificación de dígitos.
    """
    modelo = Sequential([
        Input(shape=(28 * 28,)),  # Capa de entrada con 784 nodos (28x28 píxeles aplanados)
        Dense(512, activation='relu'),  # Capa oculta con 512 neuronas y activación ReLU
        Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada dígito)
    ])
    
    modelo.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',  # Función de pérdida para clasificación multiclase
        metrics=['accuracy']  # Evaluación basada en precisión
    )
    return modelo

def entrenar_y_evaluar():
    """
    Entrena el modelo de red neuronal y lo evalúa en datos de prueba.
    """
    # Cargar y visualizar datos
    train_data_x, train_labels_y, test_data_x, test_labels_y = cargar_datos()
    print("Forma de los datos de entrenamiento:", train_data_x.shape)
    print("Etiqueta del primer ejemplo:", train_labels_y[1])
    visualizar_ejemplo(train_data_x)
    
    # Preprocesamiento de los datos
    x_train, y_train = preprocesar_datos(train_data_x, train_labels_y)
    x_test, y_test = preprocesar_datos(test_data_x, test_labels_y)
    
    # Construcción del modelo
    modelo = construir_modelo()
    print("Resumen del modelo:")
    modelo.summary()
    
    # Entrenamiento
    print("Entrenando la red neuronal...")
    modelo.fit(x_train, y_train, epochs=10, batch_size=128)
    
    # Evaluación del modelo
    print("Evaluando el modelo...")
    loss, accuracy = modelo.evaluate(x_test, y_test)
    print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")

plt.show()