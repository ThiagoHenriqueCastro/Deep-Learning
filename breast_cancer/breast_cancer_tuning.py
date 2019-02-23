from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import keras
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede(optmizer, loos, kernel_initializer, activation, neurons):
    # criacao das camadas
    classificador = Sequential()

    # camada oculta 1 (deve apresentar o input_dim <numero de entradas> )
    classificador.add(Dense(units = neurons,
                            activation = activation,
                            kernel_initializer = kernel_initializer,
                            input_dim=30))
    
    classificador.add(Dropout(0.2))
    
    # demais camadas ocultas sem o dim
    classificador.add(Dense(units = neurons,
                            activation = activation,
                            kernel_initializer = kernel_initializer))
    
    classificador.add(Dropout(0.2))
    
    # camada de saida
    classificador.add(Dense(units=1,
                            activation='sigmoid',
                            ))
    
    # otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    
    classificador.compile(optimizer = optmizer,
                          loss = loos,
                          metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
#batch_size = numero de registros antes de atualizar pesos
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optmizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy','hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tahn'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
best = grid_search.best_params_
precisao = grid_search.best_score_

