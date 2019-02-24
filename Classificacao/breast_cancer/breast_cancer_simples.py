# imports das libs
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# leitura e armazenamento da base de dados
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

# divisao da base de dados entre treinamento e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsores, classe, test_size=0.25)

# criacao das camadas
classificador = Sequential()

# camada oculta 1 (deve apresentar o input_dim <numero de entradas> )
classificador.add(Dense(units=16,
                        activation='relu',
                        kernel_initializer='random_uniform',
                        input_dim=30))

# demais camadas ocultas sem o dim
classificador.add(Dense(units=16,
                        activation='relu',
                        kernel_initializer='random_uniform'))

# camada de saida
classificador.add(Dense(units=1,
                        activation='sigmoid',
                        ))

# otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)

classificador.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])


# classificador.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                      metrics=['binary_accuracy'])

classificador.fit(previsores_treinamento,
                  classe_treinamento,
                  batch_size=10,
                  epochs=100)

pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)
