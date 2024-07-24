import cv2
import numpy as np


class PreprocessImg(object):
    """
    Recibe el arreglo que represennta la imágen y aplica procesamiento a este
    """

    @classmethod
    def preprocess(self, array):
        """Recibe el arreglo que represeta la imagen y realiza las siguientes modificaciones:
        - resize a 512x512
        - conversión a escala de grises
        - ecualización del histograma con CLAHE
        - normalización de la imagen entre 0 y 1
        - conversión del arreglo de imagen a formato de batch (tensor)


        Args:
            array (_type_): Imagen

        Returns:
            _type_: Imagen preprocesada
        """
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array
