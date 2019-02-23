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

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.015, 0.03, 0.007, 23.15, 16,64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.263]])
previsao = classificador.predict(novo)