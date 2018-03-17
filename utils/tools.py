import numpy as np
import math 
import sys

class Initializer():
    
    def __init__(self, params):
        raise NotImplementedError

    def initialize(self, size):
        raise NotImplementedError

class Guassian(Initializer):
    
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def initialize(self, size):
        return np.random.normal(self.mean, self.std, size=size)
    
class Uniform(Initializer):
    
    def __init__(self, a=-0.05, b=0.05):
        self.a = a
        self.b = b

    def initialize(self, size):
        return np.random.uniform(self.a, self.b, size=size)

class Xavier(Initializer):

    def __init__(self, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out

    def initialize(self, size):
        return np.random.normal(0, math.sqrt(2/(self.fan_in+self.fan_out)), size=size)

class MSRA(Initializer):

    def __init__(self, fan_in):
        self.fan_in = fan_in

    def initialize(self, size):
        return np.random.normal(0, math.sqrt(2/self.fan_in), size=size)

def clip_gradients(in_grads, clip=1):
    return np.clip(in_grads, -clip, clip)

def rel_error(x, y):
    rel_error = np.abs(x-y)/(np.maximum(1e-8, np.abs(x)+np.abs(y)))
    rel_error = np.sum(np.nan_to_num(rel_error))/np.sum(~np.isnan(rel_error))
    return rel_error