"""CNN image-classification model for the HandSpeak 26-class (A-Z) dataset.

This module defines a compact convolutional neural network that consumes
28x28x1 grayscale hand-sign images and outputs a probability distribution
over the 26 ASL alphabet letters (A-Z, labels 0-25).

It is intentionally kept separate from the preprocessing pipeline so that
the existing dataset/preprocessing files remain unchanged.

NOTE: Data augmentation is applied OFFLINE to the training arrays (see
``scripts/train_improved_cnn.py``), not inside this model. Keras-native
``RandomRotation``/``RandomZoom``/``RandomBrightness`` layers were found to
collapse training on the native-Windows CPU TensorFlow build, so they are
deliberately not used here.
"""

from keras import Input, Model, optimizers, regularizers
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

# 26 letters of the ASL alphabet, in model-output order (A=0 ... Z=25).
# Imported from the canonical label mapping so inference and training always
# agree on class ordering.
from utils.label_mapping import LETTERS  # noqa: E402


def build_cnn_model(input_shape=(28, 28, 1), num_classes=26, learning_rate=1e-3,
                    l2=1e-4):
    """Build and compile the HandSpeak 26-class CNN.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input image (height, width, channels).
    num_classes : int
        Number of output neurons (26 for A-Z).
    learning_rate : float
        Optimizer learning rate.
    l2 : float
        L2 weight-decay factor applied to convolutional and dense kernels to
        reduce overfitting and improve generalization to webcam frames.

    Returns
    -------
    keras.Model
        A compiled model with categorical-crossentropy loss.
    """
    kernel_reg = regularizers.l2(l2)

    inputs = Input(shape=input_shape, name="handsign_input")

    # Block 1
    x = Conv2D(32, (3, 3), padding="same", activation="relu",
               kernel_regularizer=kernel_reg)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding="same", activation="relu",
               kernel_regularizer=kernel_reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding="same", activation="relu",
               kernel_regularizer=kernel_reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Classifier head
    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=kernel_reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax", name="letter_probs")(x)

    model = Model(inputs=inputs, outputs=outputs, name="handspeak_cnn")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
