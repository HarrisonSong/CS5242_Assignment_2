import numpy as np
import importlib
import rnn_layers
importlib.reload(rnn_layers)
from rnn_layers import RNNCell, BidirectionalRNN
from utils.check_grads import check_grads_layer

N, T, D, H = 2, 3, 4, 5
x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
in_grads = np.random.uniform(size=(N, T, H*2))

rnn_cell = RNNCell(in_features=D, units=H)
brnn = BidirectionalRNN(rnn_cell)
check_grads_layer(brnn, x, in_grads)