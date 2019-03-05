import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import cv2 as cv
from PIL import Image

arquivo = open('classificador_simpsons.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_simpsons.h5')

video = cv.VideoCapture(0)


while True:
    _, frame = video.read()
    
    capture = cv.waitKey(1)
    if capture == ord('v'):
    
        cv.imwrite('frame.jpg', frame)
          
        
        imagem_teste = image.load_img('frame.jpg',
                                  target_size = (64, 64))
    
        imagem_teste = image.img_to_array(imagem_teste)
        
        imagem_teste /= 255
        
        imagem_teste = np.expand_dims(imagem_teste, axis = 0)
        
        previsao = classificador.predict(imagem_teste)

           
        if(previsao < 0.3):
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, 'Bart ', (10,450), font, 3, (0, 255, 0), 2, cv.LINE_AA)
            
        elif(previsao > 0.8):
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, 'Homer ', (10,450), font, 3, (0, 255, 0), 2, cv.LINE_AA)
        
        else:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, 'Sem identificacao '+str(previsao), (10,450), font, 3, (0, 255, 0), 2, cv.LINE_AA)
        
    cv.imshow("Capturing", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        video.release()
        cv.destroyAllWindows()
        break
        