from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import keras

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    # criacao das camadas
    classificador = Sequential()

    # camada oculta 1 (deve apresentar o input_dim <numero de entradas> )
    classificador.add(Dense(units=16,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            input_dim=30))
    
    classificador.add(Dropout(0.2))
    
    # demais camadas ocultas sem o dim
    classificador.add(Dense(units=16,
                            activation='relu',
                            kernel_initializer='random_uniform'))
    
    classificador.add(Dropout(0.2))
    
    # camada de saida
    classificador.add(Dense(units=1,
                            activation='sigmoid',
                            ))
    
    # otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    
    classificador.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    return classificador

# fn = metodo de criacao da rede
classificador = KerasClassifier(build_fn = criarRede, 
                                epochs = 100,
                                batch_size = 10)

# cv = 10(k do cross)
results = cross_val_score(estimator = classificador,
                          X = previsores,
                          y = classe,
                          cv = 10,
                          scoring = 'accuracy')

media_results = results.mean()
desvio_results = results.std()

