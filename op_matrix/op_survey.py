"""Test consistency between torch.onnx exported operators and aten operators."""

import dataclasses
import io
from typing import Iterator, List, Optional, Tuple

import onnx
import torch
import tqdm
from torch.onnx import _constants
from torch.testing._internal import common_methods_invocations

# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_MAX_OPSET

TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)


TESTED_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    # Floating types
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    # Complex types
    torch.complex32,
    torch.complex64,
    torch.complex128,
)


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


def produce_op_sample() -> Iterator[
    Tuple[common_methods_invocations.OpInfo, torch.nn.Module, tuple, torch.dtype]
]:
    """Produce samples of all operators to test."""

    op_db = common_methods_invocations.op_db
    for op_info in op_db:
        for dtype in TESTED_DTYPES:
            for sample in op_info.sample_inputs(
                device="cpu", dtype=dtype, requires_grad=False
            ):
                model = SingleOpModel(op_info.op, sample.kwargs)
                yield op_info, model, (sample.input, *sample.args), dtype


@dataclasses.dataclass
class OpTestResult:
    """Result of testing an operator."""

    opset: int
    dtype: torch.dtype
    operator: str
    exception: Optional[Exception]


def check_single_op(op_info, model, inputs, dtype, opset_version):
    # Export the model
    model_buffer = io.BytesIO()

    try:
        torch.onnx.export(
            model,
            inputs,
            model_buffer,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    except Exception as e:
        # TODO: Test in place variants as well
        return OpTestResult(
            opset=opset_version,
            dtype=dtype,
            operator=op_info.name,
            exception=e,
        )

    # Check the model with ONNX
    model_buffer.seek(0)
    onnx_model = onnx.load(model_buffer)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        return OpTestResult(
            opset=opset_version,
            dtype=dtype,
            operator=op_info.name,
            exception=e,
        )


def test_op_consistency() -> List[OpTestResult]:
    """Test that torch.onnx export produces the same results as aten."""
    all_samples = list(tqdm.tqdm(produce_op_sample()))
    for op_info, model, inputs, dtype in tqdm.tqdm(all_samples):
        for opset_version in TESTED_OPSETS:
            result = check_single_op(op_info, model, inputs, dtype, opset_version)
