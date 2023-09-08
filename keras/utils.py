# %%

from numpy import clip
from keras.models import Model
from keras.layers import Input, Dense, Dropout

# %%


def make_nn(input_shape, n_classes, neurons, dropout=None):

    if dropout:
        dropout = clip(dropout, 0, 1)

    inputs = Input(input_shape)
    x = Dense(neurons[0], activation="relu")(inputs)
    for n in neurons[1:]:
        x = Dense(n, activation="relu")(x)
        if dropout:
            x = Dropout(dropout)(x)
    outputs = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile("adam", "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model
