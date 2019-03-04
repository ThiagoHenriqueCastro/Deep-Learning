import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treinamento[2], cmap = 'gray')
plt.title('Classe '+ str(y_treinamento[2]))

# no caso os 28 sao as dimensoes das imagens e o ultimo os canais de cor (1 = escalas de cinza)
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)

previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

# 10 = numero de classes possiveis
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
# 32 = numero de detectores(64 recomendado para comecar)
# 3,3 = tamanho da matriz do feature detector
# strides = como a janela ira se mover na matriz
# input_shape = dimensoes e canais das imagens
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation = 'relu'))

# usar para deixar treinamento mais eficiente
classificador.add(BatchNormalization())

# pool_size = tamanho da janela que percorre a matriz convolucionada
classificador.add(MaxPooling2D(pool_size = (2,2)))


classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

# units = estimar pelas matrizes
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
classificador.fit(previsores_treinamento,
                  classe_treinamento,
                  batch_size = 128,
                  epochs = 5,
                  validation_data = (previsores_teste, classe_teste))

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

classificador_json = classificador.to_json()
with open('classificador_digitos.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_digitos.h5')


                      