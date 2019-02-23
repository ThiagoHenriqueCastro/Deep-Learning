import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutra_rede = arquivo.read()
arquivo.close()