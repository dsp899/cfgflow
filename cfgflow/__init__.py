import cfgflow.net
"""
pynet = cfgflow.net.PYNet()
tfnet = cfgflow.net.TFNet(pynet.init,pynet.pylayers,mode=True)
#tfnet.generate_train_sergio(learnrate=0.0005)

tfnet.generate_train(learnrate=0.0001)
tfnet.train(batchsize=64,epochs=1)
tfnet.processvar()
tfnet.savechkpt()
tfnet.savemeta()

tfnetInfer = cfgflow.net.TFNet(init,pylayers,mode=False)
tfnetInfer.savepb()
#tfnetInfer.infer()
"""
