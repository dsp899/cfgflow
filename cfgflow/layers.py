import numpy as np

class Layer(object):
    def __init__(self,init,cfglayer):
        self.init = init
        self.type = cfglayer[0]
        self.num = cfglayer[1]
        self.param_shapes = None
        if (len(cfglayer) > 2):
            self.generate(cfglayer)
   #def generate(self,cfglayer):
   #    print("soy generate")


class connected(Layer):
    def generate(self,cfglayer):
        print('soy generate de connected layer')
        print(cfglayer)

        if None in cfglayer[2][1:]:
            self.inp_shape = cfglayer[2]
        else:
            self.inp_shape = [cfglayer[2][0],np.prod(cfglayer[2][1:])]
        print("aqui: ",self.inp_shape)
        self.outputs = cfglayer[3]
        self.activation = cfglayer[4]
        self.batchnorm = cfglayer[5]
        self.param_shapes = {'biases': self.outputs, 'weights': [self.inp_shape[1], self.outputs]}

class convolution(Layer):
    def generate(self,cfglayer):
        print('soy generate de convolution layer')
        self.inp_shape = cfglayer[2]
        self.filters = cfglayer[3]
        self.ksize = cfglayer[4]
        self.stride = cfglayer[5]
        self.pad = cfglayer[6]
        self.activation = cfglayer[7]
        self.batchnorm = cfglayer[8]
        self.param_shapes = {'biases': self.filters, 'weights': [self.ksize, self.ksize, self.inp_shape[-1], self.filters]}

class maxpool(Layer):
    def generate(self,cfglayer):
        self.size = cfglayer[2]
        self.stride = cfglayer[3]
        self.pad = cfglayer[4]


pyops = dict({'connected': connected,
              'convolution': convolution,
              'maxpool': maxpool})

def create(init, cfglayer):
    print("Function create")
    layertype = cfglayer[0]
    return pyops.get(layertype,Layer)(init,cfglayer)

