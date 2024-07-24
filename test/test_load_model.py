from src.load_model import Model
import unittest


class TestImagen(unittest.TestCase):
    def setUp(self):
        self.model = Model()

    def test_load(self):
        _model = self.model.load
        self.assertNotEqual(_model, None)


if __name__ == "__main__":
    unittest.main()
