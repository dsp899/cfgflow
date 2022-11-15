import cfgflow.parser
import cfgflow.layers
import cfgflow.net

cfglayers = cfgflow.parser.cfg_yielder(__import__('__main__').cfgfile)
pylayers = list()
init = list()
for i, (cfginit, cfglayer) in enumerate(cfglayers):
    init = cfginit
    pylayer = cfgflow.layers.create(init,cfglayer)
    pylayers.append(pylayer)
tfnet = cfgflow.net.TFNet(init,pylayers,mode=True)
tfnet.generate_train(learnrate=0.0001)
tfnet.train(batchsize=64,epochs=1)
tfnet.processvar()
tfnet.savechkpt()
#tfnet.savemeta()

tfnetInfer = cfgflow.net.TFNet(init,pylayers,mode=False)
tfnetInfer.savepb()
tfnetInfer.infer()

