# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

## Autor
* Nombre: José Ángel García Gómez
* Matrícula: A01745865
* Carrera: Ingeniería en Tecnologías Computacionales
* Escuela: Tec de Monterrey, campus Estado de México

## Descripción
Un árbol de decisión es una estructura jerárquica en forma de árbol compuesta por nodos. Cada nodo interno representa una decisión basada en un atributo específico, mientras que las hojas del árbol representan las clases o etiquetas en las que se clasifican los datos. En este caso, las clases podrían ser las diferentes nacionalidades de los carros, como "estadounidense", "japonés" o "europeo".

El proceso de entrenamiento de un árbol de decisión implica dividir repetidamente los datos en función de los atributos más relevantes. El algoritmo busca la mejor manera de dividir los datos en cada nodo interno, de manera que se maximice la pureza de las clases resultantes. Este proceso continúa hasta que se alcanza un cierto criterio de parada, como la profundidad máxima del árbol o el número mínimo de muestras en un nodo.

## Ejecución del Archivo Python
A continuación se muestra un ejemplo de cómo correr el archivo Python que utiliza un árbol de decisión para determinar la nacionalidad de un carro.

* **Instalación** de Dependencias: Asegúrate de tener instaladas las bibliotecas necesarias, como scikit-learn, que proporciona implementaciones de árboles de decisión y herramientas para el aprendizaje automático en Python. Para instalar scikit-learn, ejecuta el siguiente comando en la terminal:

```
pip install -U scikit-learn
```

* **Ejecución** del Archivo: Para ejecutar el archivo Python, ejecuta el siguiente comando en la terminal:

```
python decision_tree.py
```

## Resultados
El archivo Python genera un único dato de salida: la precisión del modelo. La precisión del modelo es la proporción de predicciones correctas realizadas por el modelo. En este caso, la precisión del modelo es del 100%, lo que significa que el modelo predice correctamente la nacionalidad de todos los carros en el conjunto de datos.