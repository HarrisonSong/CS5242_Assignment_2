# import numpy as np
# import importlib
# import rnn_layers
# importlib.reload(rnn_layers)
# from rnn_layers import RNNCell
# from utils.check_grads import check_grads_layer
#
# N, D, H = 3, 10, 4
# x = np.random.uniform(size=(N, D))
# # set part of input to NaN
# # this situation will be encountered in the following work
# x[1:, :] = np.nan
# prev_h = np.random.uniform(size=(N, H))
# in_grads = np.random.uniform(size=(N, H))
# rnn_cell = RNNCell(in_features=D, units=H)
# check_grads_layer(rnn_cell, [x, prev_h], in_grads)

import numpy as np
import importlib
import rnn_layers
importlib.reload(rnn_layers)
from rnn_layers import RNNCell, RNN
from utils.check_grads import check_grads_layer
N, T, D, H = 2, 3, 4, 5

x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
in_grads = np.random.uniform(size=(N, T, H))
rnn_cell = RNNCell(in_features=D, units=H)
rnn = RNN(rnn_cell)
check_grads_layer(rnn, x, in_grads)