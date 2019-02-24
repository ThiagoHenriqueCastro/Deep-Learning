import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# formatar as classes em problemas nao binarios
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
# se espera 3 dimensoes, ja que existem 3 neuronios
#    EX
#   iris setosa 1 0 0
#   iris virginica 0 1 0
#   iris versicolor 0 0 1
classe_dummy = np_utils.to_categorical(classe)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()

# units = (entradas + saidas)/2  EX: (4 entradas + 3 saidas)/2
classificador.add(Dense(units = 4,
                        activation = 'relu',
                        input_dim = 4))
classificador.add(Dense(units = 4,
                        activation = 'relu'))

# softmax para problemas de classificacao nao binarios, com mais classes
classificador.add(Dense(units = 3,
                        activation = 'softmax'))
classificador.compile(optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')