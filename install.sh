#!/bin/bash

pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt
pip install onnx-weekly
pip install --no-deps onnxscript
