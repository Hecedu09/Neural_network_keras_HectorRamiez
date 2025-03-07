# Proyecto: Neural_network_keras_HectorRamiez

Este proyecto implementa una red neuronal con `Keras` y `TensorFlow` para la clasificación de dígitos escritos a mano utilizando el conjunto de datos MNIST. La red neuronal tiene una arquitectura de capas densas con activaciones ReLU y Softmax.

## Estructura del Proyecto
```
- src/
  - kerascode.py    # Implementación del modelo de red neuronal y funciones de entrenamiento
- main.py           # Script principal que ejecuta el entrenamiento del modelo
- README.md         # Documentación del proyecto
- .gitignore        # Archivos a ignorar por Git
- Requirements.txt  # Dependencias necesarias para ejecutar el proyecto
```

## Instalación y Uso
### 1. Clonar el Repositorio
Para obtener el código fuente, clona el repositorio en tu máquina local:
```bash
git clone https://github.com/tu-usuario/Neural_network_keras_HectorRamiez.git
cd Neural_network_keras_HectorRamiez
```

### 2. Crear un Entorno Virtual e Instalar Dependencias
Se recomienda el uso de un entorno virtual para evitar conflictos con otras bibliotecas:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r Requirements.txt
```

### 3. Ejecutar el Proyecto
Para entrenar la red neuronal y evaluar su rendimiento, ejecuta el siguiente comando:
```bash
python main.py
```
Esto cargará el dataset MNIST, entrenará la red neuronal y mostrará métricas de precisión y pérdida.

## Descripción de los Archivos
### - `main.py`
   - Punto de entrada del programa.
   - Importa la función `entrenar_y_evaluar` desde `kerascode.py` y ejecuta el entrenamiento.

### - `src/kerascode.py`
   - Contiene la implementación de:
     - Carga y preprocesamiento del conjunto de datos MNIST.
     - Construcción de la red neuronal con capas densas en Keras.
     - Función de entrenamiento y evaluación del modelo.
     - Visualización de imágenes de ejemplo del dataset.

## Objetivo y Funcionamiento del Código
Este proyecto tiene como objetivo entrenar una red neuronal simple para la clasificación de imágenes de dígitos escritos a mano.

1. Se carga el conjunto de datos MNIST.
2. Se preprocesan las imágenes y etiquetas.
3. Se construye un modelo con una capa oculta densa de 512 neuronas y activación ReLU.
4. Se entrena el modelo durante 10 épocas con lotes de 128 ejemplos.
5. Se evalúa la precisión del modelo sobre datos de prueba.

## Dependencias del Proyecto
El archivo `Requirements.txt` incluye:
```
numpy
matplotlib
tensorflow
keras
```
Asegúrate de instalarlas antes de ejecutar el proyecto.
