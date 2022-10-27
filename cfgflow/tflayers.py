import tensorflow as tf
import numpy as np

class TFLayer(object):
    def __init__(self,args):
        self.num = args[0]
        self.inp = args[1]
        self.out = None
        self.lay = args[2]
        self.scope = '{}-{}'.format(str(self.num), self.lay.type)
        if self.lay.param_shapes != None:
            self.generate_variables()
        self.out = self.generate_nodes()

    def generate_variables(self):
        print('Shape: ', self.lay.param_shapes['weights'])
        with tf.variable_scope(self.scope):
            self.weights = tf.get_variable('weights',shape=self.lay.param_shapes['weights'] ,initializer=tf.contrib.layers.xavier_initializer())
            self.biases = tf.get_variable('biases',shape=self.lay.param_shapes['biases'] ,initializer=tf.contrib.layers.xavier_initializer())
            if(self.lay.batchnorm):
                self.offset = tf.get_variable('offset',shape=tf.TensorShape(self.inp.out.shape).as_list()[1:],initializer=tf.contrib.layers.xavier_initializer())
                self.scale = tf.get_variable('scale',shape=tf.TensorShape(self.inp.out.shape).as_list()[1:],initializer=tf.contrib.layers.xavier_initializer()) 
        print(self.weights.shape)

    def generate_nodes(self):
       return self.inp


class TFConnected(TFLayer):
    def generate_nodes(self):
        #Previous node don't need reshape his output (e.g is a connected node)
        if None in self.lay.init[1:]:
            temp = sel.inp
        #Previous node need to reshape his output (e.g is a convolution node)
        else:
            print(self.inp.out.shape)
            temp = tf.reshape(self.inp.out,[-1, np.prod(tf.TensorShape(self.inp.out.shape).as_list()[1:])],name=self.scope)
            print('Weights connected layer: ',self.weights.shape)
        temp = tf.nn.xw_plus_b(x=temp, weights=self.weights, biases=self.biases, name=self.scope)
        if(self.lay.batchnorm):
            #scale = tf.constant(1.0, shape=tf.TensorShape(self.inp.out.shape).as_list()[1:],name=self.scope)
            #beta = tf.constant(0.0, shape=tf.TensorShape(self.inp.out.shape).as_list()[1:],name=self.scope)
            mean, variance = tf.nn.moments(temp,axes=[0],name=self.scope)
            mean = tf.cast(mean,tf.float32,name=self.scope)
            variance = tf.cast(variance,tf.float32,name=self.scope)
            #mean = tf.get_variable('mean',shape=mean.shape,initializer=mean,trainable=False)
            #variance = tf.get_variable('variance',shape=variance.shape,initializer=variance,trainable=False)

            temp = tf.nn.batch_normalization(temp,mean,variance,self.offset,self.scale,1e-3,name=self.scope)
        return temp


class TFConvolution(TFLayer):
    def generate_nodes(self):
        print(self.inp.out)
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]],name=self.scope)
        temp = tf.nn.conv2d(temp, self.weights, padding = 'VALID',name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])
        temp = tf.nn.bias_add(value=temp, bias=self.biases, name=self.scope)
        if(self.lay.batchnorm):
            scale = tf.constant(1.0, shape=tf.TensorShape(self.inp.out.shape).as_list()[1:],name=self.scope)
            beta = tf.constant(0.0, shape=tf.TensorShape(self.inp.out.shape).as_list()[1:],name=self.scope)
            mean, variance = tf.nn.moments(temp,axes=[0],name=self.scope)
            mean = tf.cast(mean,tf.float32,name=self.scope)
            variance = tf.cast(variance,tf.float32,name=self.scope)
            #mean = tf.get_variable('mean',initializer=mean,trainable=False)
            #variance = tf.get_variable('variance',initializer=variance,trainable=False)
            temp = tf.nn.batch_normalization(temp,mean,variance,beta,scale,1e-3,name=self.scope)
        return temp

class TFMaxpool(TFLayer):
    def generate_nodes(self):
        return tf.nn.max_pool(self.inp.out, padding = 'SAME', ksize = [1] + [self.lay.size]*2 + [1], strides = [1] + [self.lay.stride]*2 + [1], name = self.scope)

class TFRelu(TFLayer):
    def generate_nodes(self):
        return tf.nn.relu(self.inp.out,name=self.scope)


tfops = dict({'connected': TFConnected,
              'convolution': TFConvolution,
              'maxpool': TFMaxpool,
              'relu': TFRelu})

def tfcreate(args):
    tftype = args[2].type
    return tfops.get(tftype,TFLayer)(args)


