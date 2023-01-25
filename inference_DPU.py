"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import numpy as np
import xir
import vart
import os
import threading
import time
import sys
import cfgflow.dataload


def accuracy(logits,labels):
    a = np.argmax(logits)
    b = np.argmax(labels)
    acierto = 1 if a == b else 0
    return acierto


def runResnet50(runner: "Runner", images, labels, time_results):
    inputTensors = runner.get_input_tensors() #Return type list of xir.Tensor objects type
    outputTensors = runner.get_output_tensors() #Return type list of xir.Tensor objects type
    
    input_ndim = tuple(inputTensors[0].dims) #Return type list converted in tuple (3,28,28,1)
    output_ndim = tuple(outputTensors[0].dims) # Return type list converted to tuple (3,10) 
    
    count = 0
    aciertos = 0
    thw_acc = 0
    bufferOutputData = [np.empty((10000,10), dtype=np.float32, order="C")]
    while count < len(images):
        runSize = input_ndim[0] #runSize 3
        
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")] #list of numpy.ndarrays
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = images[(count + j) % len(images)].reshape(input_ndim[1:])

        thw_ini = time.time()
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
        thw_fin = time.time()

        print("Epoch: "+str(int(count/3))+", Infer "+str(images[0].shape[0])+" image(s)"+ " time: {:.2f}".format((thw_fin-thw_ini)*1000)+"ms\n")
        time_results.write("Epoch: "+str(int(count/3))+", Infer "+str(images[0].shape[0])+" image(s)"+ " time: {:.2f}".format((thw_fin-thw_ini)*1000)+"ms\n")
        thw_acc += (thw_fin - thw_ini)
        for j in range(runSize):
            bufferOutputData[0][(count+j) % len(images)] = outputData[0][j]
        count = count + runSize

    for i in range(len(images)):
        aciertos = aciertos + accuracy(bufferOutputData[0][i], labels[i])
    
    print("Aciertos DPU: ", aciertos)
    print("Tiempo: " + str(thw_acc))
    
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    timedir = os.path.join(os.path.abspath(os.path.curdir),'time_results')
    filename="{}/{}-resnet50-infer-times-1-images-10-epochs.txt".format(timedir,sys.argv[3])
    time_results = open(filename,"w")
    path = os.path.abspath(os.path.curdir) 
    x_test, y_test = cfgflow.dataload.load_mnist(path=path,kind='t10k')
    x_test = x_test[0:int(argv[4]),:]
    y_test = y_test[0:int(argv[4]),:]
    threadAll = []
    threadnum = int(argv[1])
    g = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(g)
   
    assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    
    x_test_t1 = []
    x_test_t2 = []
    y_test_t1 = []
    y_test_t2 = []
    for i in range(x_test.shape[0]):
        if(x_test.shape[0]/threadnum):
            x_test_t1.append(x_test[i:i+1,:])
            y_test_t1.append(y_test[i:i+1,:])
        else:
            x_test_t2.append(x_test[i:i+1,:])
            y_test_t2.append(y_test[i:i+1,:])

    img = []
    lbl = []
    img.append(x_test_t1)
    img.append(x_test_t2)
    lbl.append(y_test_t1)
    lbl.append(y_test_t2)

    time_start = time.time()
    for i in range(int(threadnum)):
        t = threading.Thread(target=runResnet50, args=(all_dpu_runners[i], img[i], lbl[i], time_results))
        threadAll.append(t)
     
    for x in threadAll:
        x.start()
  
    for x in threadAll:
        x.join()

    del all_dpu_runners

    time_end = time.time()
    timetotal = time_end - time_start
    print("time= ",timetotal,"seconds")
    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage : python3 resnet50.py <thread_number> <resnet50_xmodel_file>")
    else:
        main(sys.argv)
        
