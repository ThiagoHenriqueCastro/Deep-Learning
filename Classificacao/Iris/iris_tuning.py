import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

def criarRede(optmizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = 4,
                            activation = 'relu',
                            kernel_initializer = kernel_initializer,
                            input_dim = 4))
    classificador.add(Dense(units = 4,
                            activation = 'relu',
                            kernel_initializer = kernel_initializer))
    classificador.add(Dense(units = 3,
                            activation = 'softmax'))
    classificador.compile(optimizer = 'adam',
                          loss = 'sparse_categorical_crossentropy',
                          metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
#batch_size = numero de registros antes de atualizar pesos
parametros = {'batch_size': [10, 30],
              'epochs': [500, 1000],
              'optmizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy','hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tahn'],
              'neurons': [2, 4, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
best = grid_search.best_params_
precisao = grid_search.best_score_



