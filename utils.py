import os
import cfgflow.dataload

def calib_iter_mnist(iter):
    images, labels = cfgflow.dataload.load_mnist(path=os.path.abspath(os.path.curdir),kind='t10k')
    return {'input': images[iter*100:iter*100+100]}
