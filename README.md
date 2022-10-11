# `torch.onnx` op support matrix

Tool for continuously checking torch.onnx operator support status. The tool tests the torch.onnx exporter on all {torch operators, dtype} combinations and reports any errors.

Additional tools also test using `torch.fx` on the ops, as well as running the coremltools exporter for comparison.
