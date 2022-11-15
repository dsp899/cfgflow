import tensorflow as tf
import cfgflow.tflayers
import cfgflow.dataload
import os

class TFNet(object):
    def __init__(self,init,pylayers,mode):
        self.layers = pylayers
        self.tflayers = list()
        self.vardict = dict()
        self.graph = tf.Graph()
        with self.graph.as_default() as graph:
            self.inpnet = tf.placeholder(tf.float32, init, 'input')
            self.outnet = self.generate(mode)
            self.init = tf.global_variables_initializer()
            self.netvariables = tf.global_variables()
            print(tf.global_variables())

    def generate(self,mode):
        self.bnlayers = dict()
        master = None
        state = tf.identity(self.inpnet)
        for i, layer in enumerate(self.layers):
            args = [i,state,layer]
            if layer.type == 'identity':
                layer.inp1 = self.tflayers[layer.inp] 
                if (layer.out != None): 
                    layer.inp2 = self.tflayers[layer.out]
                else:
                    layer.inp2 = state.out
            elif layer.type == 'blockinit':
                layer.branch = self.tflayers[layer.inp]
                master = state.out
            elif layer.type == 'blockend':
                layer.master = master
            else: pass
            state = cfgflow.tflayers.tfcreate(args,mode)
            self.tflayers.append(state.out)
            if state.lay.type == 'batchnorm':
                self.bnlayers[state.scope] = state 
        return tf.identity(state.out, name='output')

    def generate_train(self,learnrate):
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32,shape=self.outnet.shape,name='labels')
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.outnet, onehot_labels=self.labels))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(self.loss)
            correct_prediction = tf.equal(tf.argmax(self.outnet, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.init = tf.global_variables_initializer()

    def train(self,batchsize,epochs):
        path = os.path.abspath(os.path.curdir)
        print("Starting data load")
        x_train, y_train = cfgflow.dataload.load_mnist(path=path)
        x_test, y_test = cfgflow.dataload.load_mnist(path=path,kind='t10k')
        print("Finish data load")
        total_batches = int(len(x_train)/batchsize)
        print("Starting training")
        self.trainSess = tf.Session(graph=self.graph)
        
        self.trainSess.run(self.init)
        for epoch in range(epochs):
            for i in range(total_batches):
                x_batch, y_batch = x_train[i*batchsize:i*batchsize+batchsize], y_train[i*batchsize:i*batchsize+batchsize]
                train_feed_dict = {self.inpnet:x_batch, self.labels:y_batch}
                opt = self.trainSess.run([self.optimizer],feed_dict=train_feed_dict)
            test_acc,test_loss = self.trainSess.run([self.accuracy,self.loss],feed_dict={self.inpnet: x_test,self.labels : y_test})
            print("Epoch " + str(epoch+1)+ "/"+ str(epochs) + ", Test loss= " + \
                "{:.2f}".format(test_loss) + ", Test accuracy= " + \
                "{:.2f}".format(test_acc))
        print("Finish training")

    def infer(self):
        path = os.path.abspath(os.path.curdir)
        x_test, y_test = cfgflow.dataload.load_mnist(path=path,kind='t10k')
        workspace = os.path.abspath(os.path.curdir)
        builtdir = os.path.join(workspace,'built')
        with self.graph.as_default() as graph:
            labels = tf.placeholder(tf.float32,shape=self.outnet.shape,name='labels')
            correct_prediction = tf.equal(tf.argmax(self.outnet, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            saver = tf.train.Saver()
            with tf.Session(graph=graph) as sess:
                saver.restore(sess,'{}/{}'.format(builtdir,'graph'))
                test_acc = sess.run([accuracy],feed_dict={self.inpnet: x_test,labels : y_test})
                print("Test accuracy= " + "{:.2f}".format(test_acc[0]))

    def processvar(self):
        momentdict = dict()
        for var in self.netvariables:
            node = var.name.split(':')[0]
            if 'batchnorm' in var.name:
                if 'ExponentialMovingAverage' in var.name:
                    (_type,moment,shadow) = var.name.split(':')[0].split('/')
                    lay = self.bnlayers[_type]
                    self.vardict[momentdict[moment]] = lay.moving_var[moment]
                else:
                    (_type,moment) = var.name.split(':')[0].split('/')
                    if (('mean' in moment) or ('variance' in moment)):
                        self.vardict[node] = None
                        momentdict[moment] = node
                    else:
                        self.vardict[node] = var
                    
                    
            else:
                self.vardict[node] = var
        with self.graph.as_default() as graph:
            self.saver = tf.train.Saver(var_list=self.vardict)
            print(self.vardict)


    def savechkpt(self):
        workspace = os.path.abspath(os.path.curdir)
        builtdir = os.path.join(workspace,'built')
        with self.trainSess as sess:
            self.saver.save(sess,'{}/{}'.format(builtdir,'graph'),write_meta_graph=False)

    def savemeta(self):
        workspace = os.path.abspath(os.path.curdir)
        builtdir = os.path.join(workspace,'built')
        with tf.Session(graph=self.graph) as sess:
            self.saver.export_meta_graph(filename='{}/{}'.format(builtdir,'graph.meta'))

    def savepb(self):
        workspace = os.path.abspath(os.path.curdir)
        builtdir = os.path.join(workspace,'built')
        with tf.Session(graph=self.graph) as sess:
            tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=builtdir, name='graph.pb', as_text=False)

