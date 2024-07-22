import cv2
import numpy as np


class PreprocessImg(object):
    """_summary_

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def preprocess(self, array):
        """_summary_

        Args:
            array (_type_): _description_

        Returns:
            _type_: _description_
        """
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array
