import sys
import cfgflow

tfnet = cfgflow.net.TFNet()
tfnet.pynet(sys.argv[1])
tfnet.generate(mode=False)
tfnet.savepb()
cfgflow.net.TFNet.freeze()


