import numpy as np
import keras
from keras import layers
import importlib
import rnn_layers
importlib.reload(rnn_layers)
from rnn_layers import RNNCell, RNN
from utils.tools import rel_error

N, T, D, H = 2, 3, 4, 5
x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
h0 = np.random.uniform(size=(H,))

rnn_cell = RNNCell(in_features=D, units=H)
rnn = RNN(rnn_cell, h0=h0)
out = rnn.forward(x)

keras_x = layers.Input(shape=(T, D), name='x')
keras_h0 = layers.Input(shape=(H,), name='h0')
keras_rnn = layers.RNN(layers.SimpleRNNCell(H), return_sequences=True,
                       name='rnn')(keras_x, initial_state=keras_h0)
keras_model = keras.Model(inputs=[keras_x, keras_h0],
                          outputs=keras_rnn)
keras_model.get_layer('rnn').set_weights([rnn.kernel,
                                          rnn.recurrent_kernel,
                                          rnn.bias])
keras_out = keras_model.predict_on_batch([x, np.tile(h0, (N, 1))])

print('Relative error (<1e-5 will be fine): {}'.format(rel_error(keras_out, out)))