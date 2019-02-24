from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import keras
import numpy as np

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador = Sequential()

    # camada oculta 1 (deve apresentar o input_dim <numero de entradas> )
classificador.add(Dense(units=8,
                            activation='relu',
                            kernel_initializer='normal',
                            input_dim=30))
    
classificador.add(Dropout(0.2))
    
    # demais camadas ocultas sem o dim
classificador.add(Dense(units=8,
                            activation='relu',
                            kernel_initializer='normal'))
    
classificador.add(Dropout(0.2))
    
    # camada de saida
classificador.add(Dense(units=1,
                            activation='sigmoid',
                            ))
    
    # otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    
classificador.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['binary_accuracy'])


classificador.fit(previsores,
                  classe,
                  batch_size=10,
                  epochs=100)

classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')