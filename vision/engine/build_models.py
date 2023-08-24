from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os, sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def unfreeze_model(model, unfree=False, top_layer=10):
    model.trainable = False
    if unfree:
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        top_layer *= -1
        for layer in model.layers[top_layer:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    return model


def building_models(base_model, shape_input=(224, 224, 3), num_class=2, unfree=False, top_layers=20):
    base_model = unfreeze_model(base_model, unfree, top_layers)
    inputs = tf.keras.layers.Input(shape=shape_input, name="input_layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(np.unique(num_class)), activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs, outputs)
    return model