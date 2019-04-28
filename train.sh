#!/usr/bin/env sh
set -e

/home/labseashell/caffe-master/build/tools/caffe train \
	--gpu=0 --solver=solver.prototxt \
	--weights 3_prune0.3_0.3_mobilenet_v2.caffemodel \
	2>&1 | tee -a 3_prune0.3_0.3_mobilenet_v2.log $@


