import sys
import os
import cfgflow

tfnet = cfgflow.net.TFNet()
tfnet.data_load(train=60000,test=10000)
tfnet.eval_from_pb()
tfnet.infer_from_pb(batch=1,epochs=10,filename="{}-{}-infer-times-1-images-10-epochs.txt".format(sys.argv[1],sys.argv[2]))
