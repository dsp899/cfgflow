In environment base of user alumno there is tensorflow-gpu version 1.15 installed. Furtheremore it contains u environment variable LD_LIBRARY_PATH set to /usr/local/cuda-10.0/lib64. There tensorflow-gpu framework open dinamyc librarys needed to execute his GPU version. The especial library libcudnn.so versión 7.6.5 needed from version 1.15 is located in /usr/lib/x86_64-linux-gnu/libcudnn.so. The command used for find out his location is find /usr -name libcudnn.so.

The order is documented in the manual of the dynamic linker, which is ld.so. It is:

directories from LD_LIBRARY_PATH;
directories from /etc/ld.so.conf;
/lib;
/usr/lib.
(I'm simplifying a little, see the manual for the full details.)




python3 train.py dani-0.cfg
python3 infer.py dani-0.cfg

python3 measuretime.py gpu

cd ~/DAIS/Vitis-AI/
./docker.sh xilinx/vitis-ai-cpu:1.3.411
source aibashrc
get_card() #--> 1 (U50)
conda activate vitis-ai-tensorflow # tf v1.15

cd demo/VART/dais_cfgflow/

python3 measuretime.py gpu
python xilinx.py
python inference_DPU.py 1 built/compiledGraph.xmodel dpu 30
