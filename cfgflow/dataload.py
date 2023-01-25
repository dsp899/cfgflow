import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import os

PATH='/workspace/demo/VART/yolo/data/mnist/'
def load_mnist(path=PATH, kind='train'):

    """Load MNIST data from `path`"""
    data_path = os.path.join(path,'data')
    labels_path = os.path.join(data_path,'{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(data_path,'{}-images-idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        a = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(a.shape[0],28,28,1)
    
    #labels = tf.keras.utils.to_categorical(labels,num_classes=10)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    labels=b
    return images, labels

def load_sergio(path=PATH):
    data_path = os.path.join(path,'data/papi_data_de_cuda_y_tiempo.csv')
    data = pd.read_csv(data_path ,header=0)
    numpyData = data.to_numpy()
    numero_de_contadores=17 #16 para el otro set de eventos (el v1) y 14 para la otra +1 si quiero meter el tiempo de intel
    x = numpyData[:,0:numero_de_contadores] #los valores de los contadores
    y = numpyData[:, [-1]] #consigue la ultima columna. Si lo quiero de una sola diension en vez de bidimensional: [:, -1] prediciendo arm he conseguido: La precision de esta red es: [0.31411735]
    x = normalize(x,axis=0,norm='max')
    return x, y

if __name__ == '__main__':
    x, y = load_mnist(path=os.path.abspath(os.path.curdir))
    print('Images:', x.shape, ', Labels: ', y.shape)
