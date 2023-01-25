vai_q_tensorflow quantize --input_frozen_graph built/frozenGraph.pb --input_nodes input --output_nodes output --input_fn utils.calib_iter_mnist --input_shapes ?,28,28,1 --calib_iter 10
#vai_q_tensorflow quantize --input_frozen_graph built/frozenGraph.pb --input_nodes input --output_nodes output --input_fn utils.calib_iter_mnist --input_shapes 3,28,28,1 --calib_iter 10

