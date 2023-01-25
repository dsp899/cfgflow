import tensorflow as tf
import numpy as np
import cfgflow.tflayers
import cfgflow.dataload
import cfgflow.parser
import cfgflow.layers
import os
import time


class TFNet(object):
    def __init__(self):
        self.dir = os.path.abspath(os.path.curdir)
        self.graph = tf.Graph()
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.6
    
    def pynet(self,cfgfile):
        print("Starting to create PyLayers") 
        cfglayers = cfgflow.parser.cfg_yielder("{}/{}/{}".format(self.dir,'cfg',cfgfile))
        self.pylayers = list()
        for i, (cfginit, cfglayer) in enumerate(cfglayers):
            pylayer = cfgflow.layers.create(cfginit,cfglayer)
            self.pylayers.append(pylayer)
        self.cfginit = cfginit
        print("Finish to create PyLayers")

    def generate(self,mode):
        print("Starting to create TFLayers")
        self.tflayers = list()
        with self.graph.as_default() as graph:
            if None in self.cfginit[1:]: 
               self.cfginit = self.cfginit[:2]
            self.inpnet = tf.placeholder(tf.float32, self.cfginit, 'input')
            self.init = tf.global_variables_initializer()
            self.bnlayers = dict()
            master = None
            state = tf.identity(self.inpnet)
            for i, layer in enumerate(self.pylayers):
                args = [i,state,layer]
                master = self.genbranch(master,layer,state)
                state = cfgflow.tflayers.tfcreate(args,mode)
                self.tflayers.append(state.out)
                if state.lay.type == 'batchnorm':
                    self.bnlayers[state.scope] = state 
            self.outnet = tf.identity(state.out, name='output')
            self.netvariables = tf.global_variables()
        print("Finish to create TFLayers")
    
    def genbranch(self,master,layer,state):
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
        return master

    def generate_train_sergio(self,learnrate):
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32,shape=self.outnet.shape,name='labels')
            self.loss = tf.losses.absolute_difference(labels=self.labels, predictions=self.outnet)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnrate).minimize(self.loss)
            self.accuracy = tf.metrics.mean_absolute_error(labels=self.labels,predictions=self.outnet)
            print(tf.local_variables())
            self.init_global = tf.global_variables_initializer()
            self.init_local = tf.local_variables_initializer()
    
    def generate_train(self,learnrate):
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32,shape=self.outnet.shape,name='labels')
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.outnet, onehot_labels=self.labels))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(self.loss)
            correct_prediction = tf.equal(tf.argmax(self.outnet, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.init = tf.global_variables_initializer()

    def data_load(self,train,test):
        print("Starting data load")
        self.x_train, self.y_train = cfgflow.dataload.load_mnist(path=os.path.abspath(os.path.curdir))
        self.x_test, self.y_test = cfgflow.dataload.load_mnist(path=os.path.abspath(os.path.curdir),kind='t10k')
        if(train < self.x_train.shape[0]):
            self.x_train = self.x_train[0:train,]
            self.y_train = self.y_train[0:train,]
        else:
            self.x_train = self.x_train
            self.y_train = self.y_train
        
        if(test < self.x_test.shape[0]):
            self.x_test = self.x_test[0:test,]
            self.y_test = self.y_test[0:test,]
        else: 
            self.x_test = self.x_test
            self.y_test = self.y_test 
        print("Finish data load")

    def train(self,batchsize,epochs):
        total_batches = int(len(self.x_train)/batchsize)
        print("Starting training")
        self.trainSess = tf.Session(graph=self.graph,config=self.config)
        
        self.trainSess.run(self.init)
        for epoch in range(epochs):
            for i in range(total_batches):
                x_batch, y_batch = self.x_train[i*batchsize:i*batchsize+batchsize], self.y_train[i*batchsize:i*batchsize+batchsize]
                train_feed_dict = {self.inpnet:x_batch, self.labels:y_batch}
                #start = time.time()
                train_loss,opt = self.trainSess.run([self.loss,self.optimizer],feed_dict=train_feed_dict)
                #end = time.time()
                #print("Optimizer time: ",(end-start)*1000,"ms" , "for ",self.x_train.shape[0], "images")
                if(i % 20 == 0): print("Batch: ", i, "Train Loss: ", train_loss)
            test_acc,test_loss = self.trainSess.run([self.accuracy,self.loss],feed_dict={self.inpnet: self.x_test,self.labels : self.y_test})
            print("Epoch " + str(epoch+1)+ "/"+ str(epochs) + ", Test loss= " + \
                "{:.2f}".format(test_loss) + ", Test accuracy= " + \
                "{:.2f}".format(test_acc))
        print("Finish training")

    def infer_from_pb(self,batch,epochs,filename):
        infer_out = None
        time_results = open("{}/{}/{}".format(self.dir,'times',filename),"w")
        with tf.gfile.GFile("{}/{}/{}".format(self.dir,'built','frozenGraph.pb'), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,name='frozenGraph')
            inpnet = graph.get_tensor_by_name('frozenGraph/input:0')
            outnet = graph.get_tensor_by_name('frozenGraph/output:0')
            with tf.Session(graph=graph,config=self.config) as sess:
                for epoch in range(epochs):
                    start = time.time()
                    infer_out = sess.run([outnet],feed_dict={inpnet: self.x_test[0:batch,:]})
                    end = time.time()
                    print("Epoch: "+str(epoch)+", Infer "+str(self.x_test[0:batch,:].shape[0])+" image(s)"+ " time: {:.2f}".format((end-start)*1000)+"ms")
                    time_results.write("Epoch: "+str(epoch)+", Infer "+str(self.x_test[0:batch,:].shape[0])+" image(s)"+ " time: {:.2f}".format((end-start)*1000)+"ms\n")
                time_results.close()
        return infer_out

    def eval_from_meta(self):
        with tf.Graph().as_default() as graph:
            inpnet = graph.get_tensor_by_name('input:0')
            outnet = graph.get_tensor_by_name('output:0')
            labels = tf.placeholder(tf.float32,shape=outnet.shape,name='labels')
            correct_prediction = tf.equal(tf.argmax(outnet, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            saver = tf.train.Saver()
            with tf.Session(graph=graph,config=self.config) as sess:
                saver = tf.train.import_meta_graph('{}/{}/{}'.format(self.builtdir,'graph','.meta'))
                saver.restore(sess,'{}/{}'.format(self.builtdir,'graph'))
                test_acc = sess.run([accuracy],feed_dict={self.inpnet: self.x_test,labels : self.y_test})
                print("Test accuracy= " + "{:.2f}".format(test_acc[0]))
    
    def eval_from_pb(self):
        with tf.gfile.GFile("{}/{}/{}".format(self.dir,'built','frozenGraph.pb'), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,name='frozenGraph')
            #for node in tf.get_default_graph().as_graph_def().node:
            #    print(node.name)
            inpnet = graph.get_tensor_by_name('frozenGraph/input:0')
            outnet = graph.get_tensor_by_name('frozenGraph/output:0')
            labels = tf.placeholder(tf.float32,shape=outnet.shape,name='labels')
            correct_prediction = tf.equal(tf.argmax(outnet, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            with tf.Session(graph=graph,config=self.config) as sess:
                test_acc = sess.run([accuracy],feed_dict={inpnet: self.x_test,labels : self.y_test})
        print("Test accuracy= " + "{:.2f}".format(test_acc[0]))


    def processvar(self):
        print("Starting convert variables from train model to infer model")
        momentdict = dict()
        variables = dict()
        for var in self.netvariables:
            node = var.name.split(':')[0]
            if 'batchnorm' in var.name:
                if 'ExponentialMovingAverage' in var.name:
                    (_type,moment,shadow) = var.name.split(':')[0].split('/')
                    lay = self.bnlayers[_type]
                    variables[momentdict[moment]] = lay.moving_var[moment]
                else:
                    (_type,moment) = var.name.split(':')[0].split('/')
                    if (('mean' in moment) or ('variance' in moment)):
                        variables[node] = None
                        momentdict[moment] = node
                    else:
                        variables[node] = var    
            else:
                variables[node] = var
        with self.graph.as_default() as graph:
            self.saver = tf.train.Saver(var_list=variables)
            #print(variables)
        print("Finish convert variables from train model to infer model")


    def savechkpt(self):
        with self.trainSess as sess:
            self.saver.save(sess,'{}/{}/{}'.format(self.dir,'built','graph'),write_meta_graph=False)

    def savemeta(self):
        with tf.Session(graph=self.graph) as sess:
            self.saver.export_meta_graph(filename='{}/{}/{}'.format(self.dir,'built','graph.meta'))

    def savepb(self):
        with tf.Session(graph=self.graph) as sess:
            tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir="{}/{}".format(self.dir,'built'), name='graph.pb', as_text=False)
    
    @staticmethod
    def freeze():
        os.system('{}/{}/{}'.format(os.path.abspath(os.path.curdir) ,'scripts','freeze.sh'))

    @staticmethod 
    def quantize():
        os.system('{}/{}/{}'.format(os.path.abspath(os.path.curdir) ,'scripts','quantize.sh'))

    @staticmethod 
    def compile():
        os.system('{}/{}/{}'.format(os.path.abspath(os.path.curdir),'scripts','compile.sh'))
