import tensorflow as tf
import numpy as np

class TFLayer(object):
    def __init__(self,args,train):
        self.num = args[0]
        self.inp = args[1]
        self.out = None
        self.lay = args[2]
        self.scope = '{}-{}'.format(str(self.num), self.lay.type)
        #print(self.scope)
        if self.lay.param_shapes != None:
            self.generate_variables()
        self.out = self.generate_nodes(train)

    def generate_variables(self):
        with tf.variable_scope(self.scope):
            if ('weights' or 'biases') in self.lay.param_shapes:
                self.weights = tf.get_variable('weights',shape=self.lay.param_shapes['weights'] ,initializer=tf.contrib.layers.xavier_initializer())
                self.biases = tf.get_variable('biases',shape=self.lay.param_shapes['biases'] ,initializer=tf.contrib.layers.xavier_initializer())
            if 'batchnorm' in self.lay.param_shapes:
                self.mean = tf.get_variable('mean',initializer=np.zeros(self.lay.param_shapes['batchnorm'],dtype='float32'))
                self.variance = tf.get_variable('variance',initializer=np.ones(self.lay.param_shapes['batchnorm'],dtype='float32'))
                self.beta = tf.get_variable('beta',initializer=np.zeros(self.lay.param_shapes['batchnorm'],dtype='float32'))
                self.gamma = tf.get_variable('gamma',initializer=np.ones(self.lay.param_shapes['batchnorm'],dtype='float32'))
                

    def generate_nodes(self,train):
       return self.inp


class TFConnected(TFLayer):
    def generate_nodes(self,train):
        #Previous node don't need reshape his output (e.g is a connected node)
        if None in self.lay.init[1:]:
            temp = self.inp.out
        #Previous node need to reshape his output (e.g is a convolution node)
        else:
            temp = tf.reshape(self.inp.out,[-1, np.prod(tf.TensorShape(self.inp.out.shape).as_list()[1:])],name=self.scope)
        return tf.nn.xw_plus_b(x=temp, weights=self.weights, biases=self.biases, name=self.scope)

class TFConvolution(TFLayer):
    def generate_nodes(self,train):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]],name=self.scope)
        temp = tf.nn.conv2d(temp, self.weights, padding = 'VALID',name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])
        return tf.nn.bias_add(value=temp, bias=self.biases, name=self.scope)
        
class TFBatchnorm(TFLayer):
    def __init__(self,args,train):
        if train: self.ewma_trainer = tf.train.ExponentialMovingAverage(decay=0.99)
        self.moving_var = dict()
        self.epsilon = 0.001
        self.scale_after_norm = True
        super().__init__(args,train)

    def generate_nodes(self,train):
        self.axes = [0] if ('connected' in self.inp.out.name) else [0,1,2]
        if train:
            self.update = self.ewma_trainer.apply([self.mean, self.variance])
            self.moving_var['mean'] = self.ewma_trainer.average(self.mean)
            self.moving_var['variance'] = self.ewma_trainer.average(self.variance)
            mean, variance = tf.nn.moments(self.inp.out,axes=self.axes,keep_dims=False)
            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance,self.update]):
                temp = tf.nn.batch_norm_with_global_normalization(
                        self.inp.out, mean, variance, self.beta, self.gamma,
                        self.epsilon, self.scale_after_norm,name="batchnorm")
            with tf.control_dependencies([self.update]):
                return tf.identity(temp)
        else:
            return tf.nn.batch_norm_with_global_normalization(
                    self.inp.out, self.mean, self.variance, self.beta, self.gamma,
                    self.epsilon, self.scale_after_norm,name="batchnorm")



class TFMaxpool(TFLayer):
    def generate_nodes(self,train):
        return tf.nn.max_pool(self.inp.out, padding = 'SAME', ksize = [1] + [self.lay.size]*2 + [1], strides = [1] + [self.lay.stride]*2 + [1], name = self.scope)

class TFAveragepool(TFLayer):
    def generate_nodes(self,train):
        return tf.nn.avg_pool(self.inp.out, padding = 'SAME', ksize = [1] + [self.lay.size]*2 + [1], strides = [1] + [self.lay.stride]*2 + [1], name = self.scope)

class TFRelu(TFLayer):
    def generate_nodes(self,train):
        return tf.nn.relu(self.inp.out,name=self.scope)

class TFIdentity(TFLayer):
    def generate_nodes(self,train):
        return tf.add(self.lay.inp1,self.lay.inp2,name=self.scope)

class TFBlockinit(TFLayer):
    def generate_nodes(self,train):
        return tf.identity(self.lay.branch,name=self.scope)

class TFBlockend(TFLayer):
    def generate_nodes(self,train):
        return tf.add(self.inp.out,self.lay.master,name=self.scope)

tfops = dict({'connected': TFConnected,
              'convolution': TFConvolution,
              'batchnorm': TFBatchnorm,
              'maxpool': TFMaxpool,
              'averagepool': TFAveragepool,
              'relu': TFRelu,
              'identity': TFIdentity,
              'blockinit': TFBlockinit,
              'blockend': TFBlockend})

def tfcreate(args,train):
    tftype = args[2].type
    return tfops.get(tftype,TFLayer)(args,train)


