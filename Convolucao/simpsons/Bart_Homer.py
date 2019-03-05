import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

arquivo = open('classificador_simpsons.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_simpsons.h5')

imagem_teste = image.load_img('homer.jpg',
                              target_size = (64, 64))

imagem_teste = image.img_to_array(imagem_teste)

imagem_teste /= 255

imagem_teste = np.expand_dims(imagem_teste, axis = 0)

previsao = classificador.predict(imagem_teste)

if(previsao > 0.5):
    previsao = 'Homer'
else:
    previsao = 'Bart'