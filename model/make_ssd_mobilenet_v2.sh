#! /bin/bash

./export_ssd_mobilenet_v2.sh

source snpe-1.43.0.2307/bin/envsetup.sh -t ~/.local/lib/python3.5/site-packages/tensorflow

snpe-tensorflow-to-dlc --input_network ~/ssd_mobilenet_v2_exported_graph/frozen_inference_graph.pb --input_dim Preprocessor/sub 1,300,300,3 --out_node detection_classes --out_node detection_boxes --out_node detection_scores --output_path ~/dlc_model/ssd_mobilenet_v2.dlc
