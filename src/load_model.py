import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class Model(object):
    """
    Carga el modelo, cabe a notar que este modelo debe caragrse en la carpeta models,
    este no se sube a Git dado su tamaño.
    """

    __instance = None

    def __new__(cls):
        """El modelo se implementa aplicando el patron de diseño Sigleton.
        el modelo se ee de la carpeta modelos

        Returns:
            _type_: _description_
        """
        if cls.__instance is None:
            cls.__instance = super(Model, cls).__new__(cls)

        return cls.__instance

    @property
    def load(self):
        """Retorna el modelo caraga e implementa un lazy apra evitar sobrecarga.

        Returns:
            _type_: Modelo
        """
        if not hasattr(self, "_model"):
            self._model = tf.keras.models.load_model("./models/conv_MLP_84.h5")

        return self._model
