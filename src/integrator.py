import numpy as np

from src.grad_cam import GradientCam
from src.load_model import Model
from src.preprocess_img import PreprocessImg


class Integrator(object):
    """
    Integra las clases y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
    """

    @classmethod
    def predict(self, array):
        """Ejecuta la predcio con base en la imagen enviada

        Args:
            array (_type_): Imagen leida

        Returns:
            _type_: predicción, probabilidad, imágen el mapa de calor generado por Grad-CAM..
        """
        #   1. call function to pre-process image: it returns image in batch format
        batch_array_img = PreprocessImg.preprocess(array)
        #   2. call function to load model and predict: it returns predicted class and probability
        _model = Model()
        model = _model.load
        prediction = np.argmax(model.predict(batch_array_img))
        proba = np.max(model.predict(batch_array_img)) * 100
        label = ""
        if prediction == 0:
            label = "bacteriana"
        if prediction == 1:
            label = "normal"
        if prediction == 2:
            label = "viral"
        #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
        heatmap = GradientCam.grad_cam(array)
        return (label, proba, heatmap)
