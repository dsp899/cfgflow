import sys
import cfgflow

tfnet = cfgflow.net.TFNet()
tfnet.pynet(sys.argv[1])
tfnet.generate(mode=True)
#tfnet.generate_train_sergio(learnrate=0.0005)
tfnet.generate_train(learnrate=0.0001)
tfnet.data_load(train=60000,test=10000)
tfnet.train(batchsize=64,epochs=1)
tfnet.processvar()
tfnet.savechkpt()
tfnet.savemeta()

