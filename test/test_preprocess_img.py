import unittest

import numpy as np

from src.preprocess_img import PreprocessImg
from src.read_img import Imagen


class TestPreprocessImg(unittest.TestCase):
    """
    Caso de prueba pre-procesamiento imágen
    """

    def setUp(self):
        self.preprocess = PreprocessImg()
        self.imagen = Imagen()

    def test_preprocess(self):
        # TODO: Debe seleccionar la ruta donde se encuentra la imágen
        arreglo, img = self.imagen.read_dicom_file(
            "c:\\Users\\Tatiana\\Downloads\\DICOM\\normal (3).dcm"
        )

        _parray = self.preprocess.preprocess(arreglo)
        self.assertTrue(len(_parray) > 0)
        self.assertIsInstance(_parray, (list, tuple, np.ndarray))


if __name__ == "__main__":
    unittest.main()
