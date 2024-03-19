import argparse, signal, os
import pandas as pd
import numpy as np
import random
import pickle

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

examples = '''Ejemplos de uso:\n
    INFORMACIÓN\n
    informacion sobre el csv -> python template.py -f data.csv -i\n
    informacion sobre el csv +  preprocesado -> python template.py -f data.csv -p\n
    binaria + decision tree -> python template.py -f data.csv -t target_column -c BINARY --algorithm 0\n
    binaria + prepocesado + undersampling + decision tree + hyperparametros -> python template.py -f data.csv -t target_column -p -u -c BINARY --algorithm 0 --hyper\n
    multiclase + preprocesado + oversampling + knn + hyperparametros -> python template.py -f data.csv -t target_column -p -o -c MULTICLASS --algorithm 1 --hyper\n
    '''

#ARGUMENTOS DEL PROGRAMA

parser = argparse.ArgumentParser(description=examples,formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-f', '--file', dest='input_file', required=True, help="Path del archivo csv.")
parser.add_argument('-t', '--target-column', dest='target_column', required=False, help="Nombre de la columna a predecir.")
parser.add_argument('-c', '--clasification-type', dest='clasification', required=False, help="BINARY/MULTICLASS")
parser.add_argument('-p', '--apply-preprocessing', dest='preprocessing', action='store_true', required=False, help="Preprocessing")
parser.add_argument('--algorithm', dest='algorithm', required=False, help="ML algorithm: 0 -> decision tree and 1 -> knn")
parser.add_argument('--hyper', dest='hyperparameter', action='store_true', required=False, help="Hyperparameter calculation")
parser.add_argument('--save', dest='save_model_option', action='store_true', required=False, help="Save generated model into .sav")

group = parser.add_mutually_exclusive_group(required=False)

# Agregar los argumentos al grupo
group.add_argument('-o', '--oversampling', dest='oversampling', action='store_true', help="Oversampling")
group.add_argument('-u', '--undersampling', dest='undersampling', action='store_true',  help="Undersampling")
group.add_argument('-a', '--automatic_sampling', dest='automatic_sampling', action='store_true', help="Automatic_sampling")

args = parser.parse_args()


# VARIABLES GLOBALES argumentos

file_name = args.input_file
target_column_name = args.target_column
clasification_type = args.clasification
preprocessing = args.preprocessing
algorithm = args.algorithm
hyperparameter = args.hyperparameter
save_model_option = args.save_model_option

oversampling = args.oversampling
undersampling = args.undersampling
automatic_sampling = args.automatic_sampling

# VARIABLES GLOBALES CONSTANTES

TOP_10_NOMBRES_CLASES = ['class', 'label', 'target', 'outcome', 'response', 'result', 'category', 'group', 'status', 'flag']

CLASIFICATION_TYPES = {
    0: 'BINARY',
    1: 'MULTICLASS'
}


def exit(signum, frame):
    os._exit(0)

signal.signal(signal.SIGINT, exit)


def force_exit():
    os._exit(1)


def open_file(file_name):
    codificaciones = ['utf-8', 'utf-16', 'latin-1', 'iso-8859-1']

    # Intenta abrir el archivo CSV con diferentes codificaciones
    for c in codificaciones:
        try:
            # Intenta cargar el archivo CSV con la codificación actual
            d = pd.read_csv(file_name, encoding=c)
            print(f"ENCODING -> '{c}'\n")
            return d

        except Exception as e:
            # Si hay un error, imprime el mensaje de error y prueba con la siguiente codificación
            print(f"No se pudo abrir el archivo con la codificación '{c}': {str(e)}")


def print_file_rows(df, rows):
    with pd.option_context('display.max_rows', None):
        print(df.head(rows))


def view_clasification_values(d, target_column_name, type):
    print("\n\nTIPO DE CLASIFICACION -> " + type)
    class_counts = d[target_column_name].value_counts()
    print(class_counts)


def get_clasification_type(d, column_name):
    
    global clasification_type

    num_unique_classes = d[column_name].nunique()

    # Determina si es un problema de clasificación binaria o multiclase
    if num_unique_classes == 2:
        clasification_type = CLASIFICATION_TYPES[0]
    else:
        clasification_type = CLASIFICATION_TYPES[1]


def check_missing_values(d, columna_especifica=False):

    print("\n\nMISSING VALUES\n")
    if columna_especifica:
        missing_values_columna_especifica = d[columna_especifica].isnull().sum()

        # Calcular el porcentaje de missing values en la columna específica
        num_filas = len(d)
        porcentaje_missing_columna_especifica = (missing_values_columna_especifica / num_filas) * 100

        # Imprimir la información
        print(f"Contador: {missing_values_columna_especifica}")
        print(f"Porcentaje: {porcentaje_missing_columna_especifica.round().astype(int)}%")
        return

    # Conta rl numero de missing values por columna
    missing_values_por_columna = d.isnull().sum()
    # Calcular el porcentaje de missing values en cada columna
    num_filas = len(d)
    porcentaje_missing_por_columna = (missing_values_por_columna / num_filas) * 100

    # Crear un DataFrame con ambas series de datos
    df_missing_info = pd.DataFrame({'Contador': missing_values_por_columna, 'Porcentaje': porcentaje_missing_por_columna})

    # Concatenar el nombre de la columna con la información
    missing_info_string =  df_missing_info['Contador'].astype(str) + ' -> ' + df_missing_info['Porcentaje'].round().astype(int).astype(str) + '%'

    print(missing_info_string)


def basic_info(d):

    global target_column_name
    global clasification_type

    print("INFORMACIÓN GENERAL")
    print(d.info())

    # Obtener el número de filas y columnas del DataFrame
    num_filas, num_columnas = d.shape
    print("\nNúmero total de filas:", num_filas)
    print("Número total de columnas:", num_columnas)

    # Obtener estadísticas descriptivas de las columnas numéricas
    # print("\nEstadísticas descriptivas de las columnas numéricas:")
    # print(d.describe())

    check_missing_values(d)

    # COMPROBAR LA COLUMNA A PREDECIR
    if not target_column_name:
        found_column = False
        for c in d.columns:
            column_name = c.lower()
            if column_name in TOP_10_NOMBRES_CLASES:
                found_column = True
                target_column_name = c

        if not found_column:
            print("\n\nERROR: No has proporcionado una columna objetivo para predecir. Se ha tratado de buscar una automaticamente pero sin exito.")
            force_exit()
    
    print("\n\nCOLUMNA OBJETIVO -> " + target_column_name)

    if not clasification_type:
        get_clasification_type(d, target_column_name)
    
    view_clasification_values(d, target_column_name, clasification_type)


def impute_missing_values(d, col, strategy='mean'):
    # Separar los datos en características (X) y la columna objetivo (y)
    X = d.dropna().drop(columns=[col])
    y = d.dropna()[col]

    # Crear un imputador SimpleImputer con estrategia de media
    imputer = SimpleImputer(strategy=strategy)

    X_imputed = imputer.fit_transform(X)

    # Entrenar un modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_imputed, y)

    # Predecir los valores faltantes
    X_missing = d[d.isnull().any(axis=1)].drop(columns=[col])
    X_missing_imputed = imputer.transform(X_missing)

    y_pred = model.predict(X_missing_imputed)
    y_pred_rounded = [int(round(pred)) for pred in y_pred]
    d_imputed = d.copy()
    d_imputed.loc[d_imputed[col].isnull(), col] = y_pred_rounded
    
    return d_imputed


def calculate_oversampling(target_column, X, y):
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame({target_column: y_resampled})], axis=1)


def calculate_undersampling(target_column, X, y):
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame({target_column: y_resampled})], axis=1)


def pre_procesado(d, target_column):

    print("\n\nPREPROCESADO DE LOS DATOS\n")

    # TARGET COLUMN DATA TYPE 
    # Mapea las diferentes clases con numeros enteros

    if d[target_column].dtype == 'object':
        mapping = {val: i for i, val in enumerate(d[target_column].dropna().unique(), start=1)}

        # Mapear los valores en la columna target_column usando el mapeo
        d[target_column] = d[target_column].map(mapping).astype('Int64')
    

    #MISSING VALUES

    missing_values_por_columna = d.isnull().sum()

    # Calcular el porcentaje de missing values en cada columna
    num_filas = len(d)
    porcentaje_missing_por_columna = (missing_values_por_columna / num_filas) * 100

    # Iterar sobre cada columna y aplicar la lógica según la tasa de missing values
    for columna, porcentaje in porcentaje_missing_por_columna.items():
        if porcentaje < 20:
            # Eliminar filas con missing values
            d = d.dropna(subset=[columna])
        elif porcentaje >= 20 and porcentaje <= 50:
            d = impute_missing_values(d, target_column, strategy='mean')
            # d = impute_missing_values(d, target_column, strategy='median')
            # d = impute_missing_values(d, target_column, strategy='most_frequent')
            # d = impute_missing_values(d, target_column, strategy='constant', fill_value=0)
        elif porcentaje > 50:
            # Eliminar la columna completa
            d = d.drop(columns=[columna])

    #OVERSAMPLING / UNDERSAMPLING

    if oversampling or undersampling or automatic_sampling:
        X = d.drop(columns=[target_column])
        y = d[target_column]

        if oversampling:
            d = calculate_oversampling(target_column, X, y)
        elif undersampling:
            d = calculate_undersampling(target_column, X, y)
        elif automatic_sampling:
            unique_min = min(y.value_counts())
            unique_max = max(y.value_counts())
            difference_percentage = ((unique_max - unique_min) / unique_min) * 100

            if difference_percentage > 20:
                
                if clasification_type == 'BINARY':
                    if random.random() < 0.5:
                        d = calculate_oversampling(target_column, X, y)
                    else:
                        d = calculate_undersampling(target_column, X, y)
                elif clasification_type == 'MULTICLASS':
                    pass
    
    print("Datos despues del preprocesado.")
    check_missing_values(d)
    view_clasification_values(d, target_column, 'MULTICLASS')
    return d


def get_results(modelo, algoritmo_string, X_test, y_test, y_pred, grid_search=False):

    # Construir un DataFrame con los resultados
    predictions = pd.Series(data=y_pred, index=X_test.index, name='predicted_value')
    # cols = [f'probability_of_value_{label}' for label in best_knn.classes_]
    # probabilities = pd.DataFrame(data=probas, index=X_test.index, columns=cols)

    results_test = X_test.join(predictions, how='left')
    # results_test = results_test.join(probabilities, how='left')
    results_test['TARGET'] = y_test  # Agregar etiquetas reales

    # Imprimir algunas filas del DataFrame para verificar los resultados
    print(results_test.head())

    # Calcular métricas de evaluación
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(algoritmo_string, accuracy)

    if grid_search:
        # Obtener los mejores hiperparámetros encontrados
        best_params = grid_search.best_params_
        print("Mejores hiperparámetros:", best_params)

    print("\nReporte de Clasificación:")
    print(class_report)

    print("\nMatriz de Confusión:")
    print(conf_matrix)

    if save_model_option:
        nombre_modelo = "model.sav"
        pickle.dump(modelo, open(nombre_modelo, 'wb'))
        print("Modelo KNN guardado correctamente como:", nombre_modelo)


def entrenar_modelo_knn(data, target_column, hyperparameter_calculation=False):

    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'n_neighbors': [3, 5],  # Prueba con estos valores de k
        'weights': ['uniform', 'distance'],  # Prueba con ambos esquemas de peso
        'p': [1, 2]  # Prueba con diferentes métricas de distancia (1: Manhattan, 2: Euclidiana)
    }

    # KNN

    X = data.drop(columns=[target_column]) 
    y = data[target_column]     

    # Convertir las columnas object a float si es necesario
    X = X.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo KNN
    knn = KNeighborsClassifier()

    if hyperparameter_calculation:

        # Configurar la búsqueda de hiperparámetros con validación cruzada
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

        # Entrenar el modelo con la busqueda de hiperparámetros
        grid_search.fit(X_train, y_train)

        # Obtener el mejor modelo encontrado por la búsqueda de hiperparámetros
        best_knn = grid_search.best_estimator_

        # Predecir las etiquetas para el conjunto de prueba utilizando el mejor modelo
        y_pred = best_knn.predict(X_test)

        get_results(best_knn, "Precisión del modelo KNN:", X_test, y_test, y_pred, grid_search=grid_search)
    else:
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        get_results(knn, "Precisión del modelo del arbol de decision:", X_test, y_test, y_pred, grid_search=None)


def entrenar_modelo_decision_tree(data, target_column, hyperparameter_calculation=False):
    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'max_depth': [None, 5, 10, 20],  # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4]    
    }

    # Decision Tree
    X = data.drop(columns=[target_column])  # olcumnas que no hay que predecir
    y = data[target_column]                  # columna a predecir

    # Convertir las columnas object a float si es necesario
    X = X.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo de árbol de decisión
    decision_tree = DecisionTreeClassifier()

    if hyperparameter_calculation:
        # Configurar la búsqueda de hiperparámetros con validación cruzada
        grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')

        # Entrenar el modelo con la búsqueda de hiperparámetros
        grid_search.fit(X_train, y_train)

        # Obtener el mejor modelo encontrado por la búsqueda de hiperparámetros
        best_decision_tree = grid_search.best_estimator_

        # Predecir las etiquetas para el conjunto de prueba utilizando el mejor modelo
        y_pred = best_decision_tree.predict(X_test)

        get_results(best_decision_tree, "Precisión del modelo del arbol de decision:", X_test, y_test, y_pred, grid_search=grid_search)
    else:
        # Entrenar el modelo sin la búsqueda de hiperparámetros
        decision_tree.fit(X_train, y_train)

        # Predecir las etiquetas para el conjunto de prueba
        y_pred = decision_tree.predict(X_test)

        get_results(decision_tree, "Precisión del modelo del arbol de decision:", X_test, y_test, y_pred, grid_search=None)


if __name__ == '__main__':

    #Cargar archivo csv
    d = open_file(file_name)

    #Analisis basico
    # print_file_rows(d, len(d))
    basic_info(d)

    # basic pre procesing
    if preprocessing:
        d = pre_procesado(d, target_column_name)

    # 0 -> decision tree
    # 1 -> knn
    if algorithm == '0':
        entrenar_modelo_decision_tree(d, target_column_name, hyperparameter_calculation=hyperparameter)
    elif algorithm == '1':
        entrenar_modelo_knn(d, target_column_name, hyperparameter_calculation=hyperparameter)