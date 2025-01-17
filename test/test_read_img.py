from src.read_img import Imagen
import unittest
from PIL import Image


class TestImagen(unittest.TestCase):
    """
    Caso de prueba lectura imágen
    """

    def setUp(self):
        self.imagen = Imagen()

    def test_read_dicom_file(self):
        # TODO: Debe seleccionar la ruta donde se encuentra la imágen
        arreglo, img = self.imagen.read_dicom_file(
            "c:\\Users\\Tatiana\\Downloads\\DICOM\\normal (3).dcm"
        )

        self.assertTrue(len(arreglo) > 0)
        self.assertIsInstance(img, Image.Image)


if __name__ == "__main__":
    unittest.main()
