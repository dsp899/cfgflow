import tensorflow as tf
import cfgflow.tflayers
import os

class TFNet(object):
    def __init__(self,init,pylayers):
        self.layers = pylayers
        self.graph = tf.Graph()
        self.init_op = None
        with self.graph.as_default() as graph:
            self.inp_net = tf.placeholder(tf.float32, init, 'input')
            self.out_net = self.generate()
            #self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            for var in tf.get_default_graph().as_graph_def().node:
                print(var.name)

    def generate(self):
        state = tf.identity(self.inp_net)
        for i, layer in enumerate(self.layers):
            args = [i,state,layer]
            state = cfgflow.tflayers.tfcreate(args)
            print('Created layer number, ', i)
        return tf.identity(state.out, name='output')

    def savepb(self):
        workspace = os.path.abspath(os.path.curdir)
        cfgdir = os.path.join(workspace,'built')
        with tf.Session(graph=self.graph) as sess:
            self.saver.export_meta_graph(filename='{}/{}'.format(cfgdir,'graph.meta'))
