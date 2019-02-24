import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

novo = np.array([[5.1, 3.5, 1.4, 0.2]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)