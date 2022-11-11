import tensorflow as tf
import numpy as np
import os

PATH='/workspace/demo/VART/yolo/data/mnist/'
def load_mnist(path=PATH, kind='train'):

    """Load MNIST data from `path`"""
    data_path = os.path.join(path,'data')
    labels_path = os.path.join(data_path,'{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(data_path,'{}-images-idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(labels.shape[0],28,28,1)
    

    labels = tf.keras.utils.to_categorical(labels,num_classes=10)

    return images, labels

if __name__ == '__main__':
    x, y = load_mnist(path=os.path.abspath(os.path.curdir))
    print('Images:', x.shape, ', Labels: ', y.shape)
