import unittest

from read_img import Imagen


class TestImagen(unittest.TestCase):
    def test_read_dicom_file(self):
        a, b = Imagen.read_dicom_file("")
