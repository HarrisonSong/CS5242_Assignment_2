import numpy as np
import keras
from keras import layers
import importlib
import rnn_layers
importlib.reload(rnn_layers)

from rnn_layers import RNNCell
from utils.tools import rel_error
N, D, H = 3, 10, 4

x = np.random.uniform(size=(N, D))
x[1:, :] = np.nan
prev_h = np.random.uniform(size=(N, H))

rnn_cell = RNNCell(in_features=D, units=H)
out = rnn_cell.forward([x, prev_h])
# compare with the keras implementation
keras_x = layers.Input(shape=(1, D), name='x')
keras_prev_h = layers.Input(shape=(H,), name='prev_h')
keras_rnn = layers.RNN(layers.SimpleRNNCell(H), name='rnn')(keras_x, initial_state=keras_prev_h)
keras_model = keras.Model(inputs=[keras_x, keras_prev_h], outputs=keras_rnn)
keras_model.get_layer('rnn').set_weights([rnn_cell.kernel,
rnn_cell.recurrent_kernel, rnn_cell.bias])
keras_out = keras_model.predict_on_batch([x[:, None, :], prev_h])
print('Relative error (<1e-6 will be fine): {}'.format(rel_error(keras_out, out)))