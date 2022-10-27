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
tfnet = cfgflow.net.TFNet(init,pylayers)
print(tfnet.out_net.shape)
tfnet.savepb()

