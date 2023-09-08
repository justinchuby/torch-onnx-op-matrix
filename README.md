# `torch.onnx` op support matrix

[![CI](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/main.yml/badge.svg)](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/main.yml)
[![Test](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/test.yml/badge.svg)](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/test.yml)
[![CoreML Tools](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/coremltools.yml/badge.svg)](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/coremltools.yml)
[![Deployment](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/deploy-react.yml/badge.svg)](https://github.com/justinchuby/torch-onnx-op-matrix/actions/workflows/deploy-react.yml)

`^` Obtain the data here under `artifacts`


Tool for continuously checking `torch.onnx` operator support status. The tool tests the `torch.onnx.{export, dynamo_export}` exporter on all {torch operators, dtype} combinations and reports any errors.

Additional tools also test running `torch.jit` on the ops, as well as running the `coremltools` exporter for comparison.


## Development

Visualization under `op-vis/`

```sh
npm install --legacy-peer-deps
```
