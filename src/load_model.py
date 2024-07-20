import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class Model:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Model, cls).__new__(cls)

        return cls.__instance

    @property
    def load(self):
        if not hasattr(self, "_model"):
            self._model = tf.keras.models.load_model("./models/conv_MLP_84.h5")

        return self._model
