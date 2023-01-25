#vai_c_tensorflow -f /workspace/demo/VART/mnist/quantize_results/quantize_eval_model.pb -a /opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json -o /workspace/demo/VART/dais_cfgflow/built/ -n compiledGraph

#vai_c_tensorflow -f /workspace/demo/VART/mnist/quantize_results/quantize_eval_model.pb -a /opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json -o /workspace/demo/VART/mnist/compile_results -n mnistU50
xnnc-run --type tensorflow --layout NHWC --model quantize_results/quantize_eval_model.pb --out built/graph.xmodel --inputs-shape 1,28,28,1
xcompiler -i built/graph.xmodel -o built/compiledGraph.xmodel -t DPUCAHX8H_ISA2_ELP2
