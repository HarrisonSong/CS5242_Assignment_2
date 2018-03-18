# import numpy as np
# import keras
# from keras import layers
# import importlib
# import rnn_layers
# importlib.reload(rnn_layers)
#
# from rnn_layers import RNNCell
# from utils.tools import rel_error
# N, D, H = 3, 10, 4
#
# x = np.random.uniform(size=(N, D))
# x[1:, :] = np.nan
# prev_h = np.random.uniform(size=(N, H))
#
# rnn_cell = RNNCell(in_features=D, units=H)
# out = rnn_cell.forward([x, prev_h])
# # compare with the keras implementation
# keras_x = layers.Input(shape=(1, D), name='x')
# keras_prev_h = layers.Input(shape=(H,), name='prev_h')
# keras_rnn = layers.RNN(layers.SimpleRNNCell(H), name='rnn')(keras_x, initial_state=keras_prev_h)
# keras_model = keras.Model(inputs=[keras_x, keras_prev_h], outputs=keras_rnn)
# keras_model.get_layer('rnn').set_weights([rnn_cell.kernel,
# rnn_cell.recurrent_kernel, rnn_cell.bias])
# keras_out = keras_model.predict_on_batch([x[:, None, :], prev_h])
# print('Relative error (<1e-6 will be fine): {}'.format(rel_error(keras_out, out)))

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
h0 = np.random.uniform(size=(N, H))
rnn_cell = RNNCell(in_features=D, units=H)
rnn = RNN(rnn_cell, h0=h0)
out = rnn.forward(x)
keras_x = layers.Input(shape=(T, D), name='x')
keras_h0 = layers.Input(shape=(H,), name='h0')
keras_rnn = layers.RNN(layers.SimpleRNNCell(H), return_sequences=True, name='rnn')(keras_x, initial_state=keras_h0)
keras_model = keras.Model(inputs=[keras_x, keras_h0], outputs=keras_rnn)
keras_model.get_layer('rnn').set_weights([rnn.kernel, rnn.recurrent_kernel, rnn.bias])
keras_out = keras_model.predict_on_batch([x, h0])
print('Relative error (<1e-6 will be fine): {}'.format(rel_error(keras_out, out)))
