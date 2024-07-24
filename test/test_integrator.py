import unittest

import numpy as np

from src.integrator import Integrator
from src.read_img import Imagen


class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = Integrator()
        self.imagen = Imagen()

    def test_predict(self):
        arreglo, img = self.imagen.read_dicom_file(
            "c:\\Users\\Tatiana\\Downloads\\DICOM\\normal (2).dcm"
        )

        label, proba, heatmap = Integrator.predict(arreglo)
        self.assertEqual(label, "normal")
        self.assertEqual(np.round(proba, 2), 99.83)

        # self.assertIsInstance(_parray, (list, tuple, np.ndarray))


if __name__ == "__main__":
    unittest.main()
