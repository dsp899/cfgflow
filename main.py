import os
import sys

#if __name__ == '__main__':

workspace = os.path.abspath(os.path.curdir)
cfgdir = os.path.join(workspace,'cfg')   #[int(sys.argv[1])]
cfgfile = os.path.join(cfgdir,sys.argv[1])
import cfgflow



