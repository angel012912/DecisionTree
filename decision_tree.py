import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools

# Se define la funcion de cost loss para identificar la puresa de una serie de datos categóricos.
def func_entropia(serie_categorica: pd.Series) -> float:
  '''
  Dada una serie de pandas, calcula la entropia.
  serie: Serie de pandas.
  Retorna: Entropia de la serie.
  '''
  if isinstance(serie_categorica, pd.Series):
    pi = serie_categorica.value_counts()/serie_categorica.shape[0] # Se calcula la probabilidad de cada valor
    valor_entropia = np.sum(-pi*np.log2(pi+1e-9)) # Se agrega un valor pequeño para evitar el log(0)
    return valor_entropia # Se retorna el valor de la entropia
  else:
    raise('Error: La variable no es una serie de pandas.')

# Se calcula la varianza como función de cost loss para identificar la pureza de una serie de datos numéricos.
def variance(serie_numerica: pd.Series) -> float:
  '''
  Función para calcular la varianza evitando nan.
  serie: variable para calcular la varianza. Debe ser una serie de pandas.
  '''
  if(len(serie_numerica) == 1): # Si la serie tiene un solo valor, la varianza es 0
    return 0
  else:
    return serie_numerica.var()

# Se define la funcion de ganancia de información como heurística para la selección de la mejor variable de split
def func_information_gain(serie_objetivo: pd.Series, decision_division: pd.Series, func=func_entropia) -> float:
  '''
  Calcula la ganancia de información de una variable dado un criterio de división.
  serie_objetivo: Serie de pandas con la variable objetivo.
  decision_division: Serie de pandas con la variable de decisión.
  func: Función de cost loss a utilizar.
  '''
  
  valores_cumplen = sum(decision_division) # Cantidad de valores que cumplen la condición
  valores_no_cumplen = decision_division.shape[0] - valores_cumplen # Cantidad de valores que no cumplen la condición
  
  if(valores_cumplen == 0 or valores_no_cumplen ==0): # Si alguno de los valores es 0, la ganancia de información es 0
    information_gain_value = 0
  
  else:
    if serie_objetivo.dtypes != 'O': # Si la variable objetivo es numérica, se calcula la varianza
      information_gain_value = variance(serie_objetivo) - (valores_cumplen/(valores_cumplen+valores_no_cumplen)* variance(serie_objetivo[decision_division])) - (valores_no_cumplen/(valores_cumplen+valores_no_cumplen)*variance(serie_objetivo[-decision_division]))
    else: # Si la variable objetivo es categórica, se calcula la entropía
      information_gain_value = func(serie_objetivo) - (valores_cumplen/(valores_cumplen+valores_no_cumplen)* func(serie_objetivo[decision_division])) - (valores_no_cumplen/(valores_cumplen+valores_no_cumplen)*func(serie_objetivo[-decision_division]))
  
  return information_gain_value

# Se define la función para obtener todas las combinaciones posibles de una variable categórica
def categorical_options(serie_categorica: pd.Series) -> list:
  '''
  Crea todas las combinaciones posibles de una serie de pandas que contiene variables categóricas.
  serie_categorica: Serie de pandas con variables categóricas.
  '''
  serie_categorica = serie_categorica.unique()
  opciones = []
  for indice in range(0, len(serie_categorica)+1): # Se recorren todos los indices posibles de la serie
      for subset in itertools.combinations(serie_categorica, indice): # Se obtienen todas las combinaciones posibles de la serie para cada indice
          subset = list(subset) # Se convierte el objeto en una lista
          opciones.append(subset) # Se agrega la lista a la lista de opciones

  return opciones[1:-1] # Se retorna la lista de opciones sin el primer y último elemento

# Se define la función para obtener la mejor división de una variable predictora
def max_information_gain_split(serie_prediccion: pd.Series, serie_objetivo: pd.Series, func=func_entropia) -> tuple:
  '''
  Dada una variable predictora y una variable objetivo, retorna la mejor división, el error y el tipo de variable basado en una función de cost loss seleccionada.
  serie_prediccion: Variable predictora como serie de pandas.
  serie_objetivo: Variable objetivo como serie de pandas.
  func: Función de cost loss a utilizar.
  '''

  split_values = [] # Se inicializa la lista de valores de división
  information_gain_values = []  # Se inicializa la lista de ganancias de información

  flag_numerica = True if serie_prediccion.dtypes != 'O' else False # Se obtiene el tipo de variable

  # Se obtienen las opciones de división de la variable según su tipo
  if flag_numerica:
    opciones = serie_prediccion.sort_values().unique()[1:] # Se ordenan los valores de la variable y se obtienen las opciones de división de la variable numérica
  else: 
    opciones = categorical_options(serie_prediccion) # Se obtienen las opciones de división de la variable categórica

  # Se recorren todas las opciones de división para calcular la ganancia de información de cada una
  for opcion in opciones:
    serie_evaluada = serie_prediccion < opcion if flag_numerica else serie_prediccion.isin(opcion)
    information_gain_value = func_information_gain(serie_objetivo, serie_evaluada, func) # Se calcula la ganancia de información de la serie evaluada con la opción de división
    information_gain_values.append(information_gain_value)
    split_values.append(opcion)

  # Se verifica que la ganancia de información sea mayor a 0
  if len(information_gain_values) == 0:
    return (None, None, None, False) # Si no hay ganancia de información, se retorna None para indicar que no se puede hacer la división
  else: # Si hay ganancia de información, se retorna la mejor división
    mejor_information_gain = max(information_gain_values) # Se obtiene la mejor ganancia de información
    index = information_gain_values.index(mejor_information_gain)
    mejor_division = split_values[index] # Se obtiene la mejor división
    return (mejor_information_gain, mejor_division, flag_numerica, True)

# Se define la función para obtener la mejor variable predictora de un data set para predecir una columna objetivo 
def get_best_split(nombre_col_objetivo: str, dataSet: pd.DataFrame) -> tuple:
  '''
  Dada una variable objetivo y un dataframe, retorna la mejor división, el error y el tipo de variable basado en una función de cost loss seleccionada.
  nombre_col_objetivo: Nombre de la variable objetivo.
  dataSet: Dataframe con las variables predictoras y la variable objetivo.
  '''
  # Se obtienen las mejores divisiones de cada variable predictora
  data_evaluada = dataSet.drop(nombre_col_objetivo, axis= 1).apply(max_information_gain_split, serie_objetivo = dataSet[nombre_col_objetivo])
  
  # Se verifica que la división sea posible
  if sum(data_evaluada.loc[3,:]) == 0:
    return (None, None, None, None)
  else:
    data_evaluada = data_evaluada.loc[:,data_evaluada.loc[3,:]]
    mejor_variable_division = data_evaluada.iloc[0].astype(np.float32).idxmax()
    mejor_valor_division = data_evaluada[mejor_variable_division][1] 
    information_gain_divison = data_evaluada[mejor_variable_division][0]
    division_numeric_flag = data_evaluada[mejor_variable_division][2]

    return (mejor_variable_division, mejor_valor_division, information_gain_divison, division_numeric_flag)

# Se define la función para dividir un data set en dos data sets según una variable predictora y un valor de división
def make_split(split_variable: str, split_value, data_to_split, numeric_flag: bool) -> tuple:
  '''
  Dado un data set y una condición de división, retorna los dos data sets divididos.
  split_variable: variable con la que se hace la división.
  split_value: valor de la variable con la que se hace la división.
  data: data a dividir.
  is_numeric: booleano que indica si la variable es numérica o no.
  '''
  if numeric_flag: # Si la variable es numérica, se hace la división con el valor de la variable
    left_data = data_to_split[data_to_split[split_variable] < split_value]
    right_data = data_to_split[(data_to_split[split_variable] < split_value) == False]
  else: # Si la variable es categórica, se hace la división con los valores de la variable
    left_data = data_to_split[data_to_split[split_variable].isin(split_value)]
    right_data = data_to_split[(data_to_split[split_variable].isin(split_value)) == False]

  return (left_data, right_data) # Se retorna los dos data sets divididos

# Se define la función para hacer una predicción de una variable objetivo dado un data set
def make_prediction(data_objetivo: pd.Series, factor_flag: bool) -> float:
  '''
  Dado una serie de datos, retorna la predicción de la variable objetivo.
  data_objetivo: Serie de pandas con la variable objetivo.
  factor_numeric: Booleano que indica si la variable objetivo es numérica o no.
  '''
  if factor_flag: # Si la variable es factor se retorna la moda
    pred_val = data_objetivo.value_counts().idxmax()
  else: # Si la variable es numérica se retorna la media
    pred_val = data_objetivo.mean()
  return pred_val

# Se define la función para entrenar un árbol de decisión con un data set y una variable objetivo
def generar_arbol(dataset: pd.DataFrame, nombre_variable_objetivo: str, factor_flag: bool, max_depth = None, min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20) -> dict:
  '''
  Entrena un árbol de decisión.
  dataset: Data set con las variables predictoras y la variable objetivo.
  nombre_variable_objetivo: Nombre de la variable objetivo.
  factor_flag: Booleano que indica si la variable objetivo es numérica o no.
  Hiperparámetros:
    max_depth: Máxima profundidad del árbol.
    min_samples_split: Minimo número de observaciones para hacer una división.
    min_information_gain: Mínima ganancia de información para hacer una división.
    max_categories: Máximo número de categorías para una variable categórica.
  '''

  # Si es la primera iteración, se verifica que el número de categorías sea válido
  if counter==0:
    types = dataset.dtypes # Se obtienen los tipos de variables
    check_columns = types[types == "object"].index # Se obtienen las variables categóricas
    for column in check_columns: # Se recorren las variables categóricas
      var_length = len(dataset[column].value_counts()) 
      if var_length > max_categories: # Se verifica que el número de categorías sea menor al máximo
        raise ValueError('Error: La variable ' + column + ' tiene '+ str(var_length) + ' categorías, que es mayor al máximo permitido: ' +  str(max_categories))

  # Se verifica que la profundidad máxima sea válida
  if max_depth == None:
    depth_flag = True # Si no hay profundidad máxima, se cumple la condición
  else: # Si hay profundidad máxima, se verifica que no se haya alcanzado
    if counter < max_depth: # Si no se ha alcanzado la profundidad máxima, se cumple la condición
      depth_flag = True
    else: # Si se ha alcanzado la profundidad máxima, no se cumple la condición
      depth_flag = False

  # Se verifica que el número de observaciones mínimas sea válido
  if min_samples_split == None:
    sample_cond = True # Si no hay número de observaciones mínimas, se cumple la condición
  else: # Si hay número de observaciones mínimas, se verifica que se cumpla
    if dataset.shape[0] > min_samples_split: # Si se cumple, se cumple la condición
      sample_cond = True 
    else: # Si no se cumple, no se cumple la condición
      sample_cond = False

  # Se verifica que la ganancia de información mínima sea válida
  if depth_flag & sample_cond: 
    variable_division, valor_division, information_gain_division, numeric_flag = get_best_split(nombre_variable_objetivo, dataset)

    # Si se cumple la condición de ganancia de información, se hace la división
    if information_gain_division is not None and information_gain_division >= min_information_gain:
      counter += 1 # Se aumenta el contador de profundidad
      data_left, data_right = make_split(variable_division, valor_division, dataset, numeric_flag)
      # Se define la condición en la que se hace la división en el árbol
      tipo_divison = "<=" if numeric_flag else "in"
      nodo =   "{} {}  {}".format(variable_division, tipo_divison, valor_division)
      arbol = {nodo: []}

      # Se obtienen las respuestas (recursión)
      subnodos_izquierdo = generar_arbol(data_left, nombre_variable_objetivo, factor_flag, max_depth, min_samples_split, min_information_gain, counter)
      subnodos_derecho = generar_arbol(data_right, nombre_variable_objetivo, factor_flag, max_depth, min_samples_split, min_information_gain, counter)

      if subnodos_izquierdo == subnodos_derecho:
        arbol = subnodos_izquierdo

      else:
        arbol[nodo].append(subnodos_izquierdo)
        arbol[nodo].append(subnodos_derecho)

    # Si no se cumple la condición de ganancia de información, se hace la predicción
    else:
      prediccion = make_prediction(dataset[nombre_variable_objetivo], factor_flag)
      return prediccion
  # Si no se cumple la condición de profundidad o de número de observaciones, se hace la predicción
  else:
    prediccion = make_prediction(dataset[nombre_variable_objetivo], factor_flag)
    return prediccion

  return arbol

# Se define la función para clasificar una observación dado un árbol de decisión
def clasificar_datos(observacion: pd.Series, arbol: dict):
  '''
  Dada una observación y un árbol de decisión, retorna la predicción de la variable objetivo.
  observacion: Serie de pandas con la observación.
  arbol: Árbol de decisión.
  '''
  
  nodo = list(arbol.keys())[0] # Se obtiene el nodo del árbol

  if nodo.split()[1] == '<=': # Se verifica si la división es numérica
    if observacion[nodo.split()[0]] <= float(nodo.split()[2]): # Se verifica si la observación cumple la condición
      subnodo = arbol[nodo][0] # Se va al nodo izquierdo
    else:
      subnodo = arbol[nodo][1] # Se va al nodo derecho
  else: # Si la división es categórica
    if observacion[nodo.split()[0]] in (nodo.split()[2]): # Se verifica si la observación cumple la condición
      subnodo = arbol[nodo][0] # Se va al nodo izquierdo
    else:
      subnodo = arbol[nodo][1] # Se va al nodo derecho

  # Si la respuesta no es un diccionario, se retorna la respuesta
  if not isinstance(subnodo, dict):
    return subnodo # Se retorna la respuesta
  else: # Si la respuesta es un diccionario, se hace recursión
    return clasificar_datos(observacion, subnodo)

# Se define la función para obtener las predicciones de un data set dado un árbol de decisión
def get_y_pred_series(dataset: pd.DataFrame, arbol: dict) -> pd.Series:
    '''
    Dado un data set y un árbol de decisión, retorna la predicción de la variable objetivo.
    data: Data set con las variables predictoras y la variable objetivo.
    arbol: Árbol de decisión.
    '''

    predicciones = []
    for index in range(dataset.shape[0]):
        predicciones.append(clasificar_datos(dataset.iloc[index], arbol))
    return(pd.Series(predicciones))

# Se define la función para obtener la precisión de un árbol de decisión
def get_precision(X_validation: pd.DataFrame, Y_validation: pd.Series, X_test: pd.DataFrame, Y_test: pd.Series, arbol: dict) -> float:
    '''
    Dado un data set y un árbol de decisión, retorna la precisión del árbol.
    X_validation: Data set con las variables predictoras de validación.
    Y_validation: Serie de pandas con la variable objetivo de validación.
    X_test: Data set con las variables predictoras de test.
    Y_test: Serie de pandas con la variable objetivo de test.
    arbol: Árbol de decisión.
    '''
    
    Y_Pred_test = get_y_pred_series(X_test, arbol)
    Y_Pred_validation = get_y_pred_series(X_validation, arbol)
    precision_test = accuracy_score(Y_test, Y_Pred_test)
    precision_validation = accuracy_score(Y_validation, Y_Pred_validation)
    mean_precision = (precision_test + precision_validation)/2
    return mean_precision

if __name__ == '__main__':
    # Leer el archivo de datos
    col_names = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year', 'brand']
    df = pd.read_csv("cars.csv", na_values=['',' '], skiprows=1, header=None, names=col_names)

    # Se eliminan los valores nulos
    df = df.dropna()

    # Se obtienen los data frames de entrenamiento y de test
    X = df.drop('brand', axis=1)
    Y = df['brand']

    # Se genera un árbol de decisión que se entrena con un data frame de entraniento random para poder identificar que el árbol generaliza y no se sobreajusta
    for _ in range(10):
      
      # Se obtiene un numero random del 1 al 100 para dividir el data set
      random_state = np.random.randint(1, 100)

      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

      # El data frame de entrenamiento se vuelve a dividir entre entrenamiento y validación
      X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)

      # Se definen los hiperparámetros
      max_depth = 10
      min_samples_split = 5
      min_information_gain  = 1e-5
      Data_train = pd.concat([X_train, Y_train], axis=1) # Se unen las variables predictoras y la variable objetivo en un solo data frame

      arbol = generar_arbol(Data_train, 'brand', True, max_depth, min_samples_split, min_information_gain) # Se entrena el árbol de decisión
      precision = get_precision(X_validation, Y_validation, X_test, Y_test, arbol) # Se obtiene la precisión del árbol de decisión
      print('La precisión del árbol de decisión es: ' + str(precision)) # Se imprime la precisión del árbol de decisión