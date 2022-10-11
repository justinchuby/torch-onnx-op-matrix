# `torch.onnx` op support matrix

[![CI](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/main.yml/badge.svg)](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/main.yml) [![CoreML Tools](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/coremltools.yml/badge.svg)](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/coremltools.yml)  
`^` Obtain the data here under `artifacts`


Tool for continuously checking torch.onnx operator support status. The tool tests the torch.onnx exporter on all {torch operators, dtype} combinations and reports any errors.

Additional tools also test using `torch.fx` on the ops, as well as running the coremltools exporter for comparison.
