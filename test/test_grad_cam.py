import unittest

from PIL import Image

from src.grad_cam import GradientCam
from src.read_img import Imagen


class TestGradcam(unittest.TestCase):
    def setUp(self):
        self.gradcam = GradientCam()
        self.imagen = Imagen()

    def test_gradcam(self):
        arreglo, img = self.imagen.read_dicom_file(
            "c:\\Users\\Tatiana\\Downloads\\DICOM\\normal (2).dcm"
        )

        heatmap = GradientCam.grad_cam(arreglo)

        self.assertTrue(len(heatmap) > 0)


if __name__ == "__main__":
    unittest.main()
