"""Test consistency between torch.onnx exported operators and aten operators."""

import itertools
import json
import dataclasses
import io
import traceback
from typing import Any, Dict, Iterator, List, Optional, Tuple

import onnx
import torch
import tqdm
from torch.onnx import _constants
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo import definitions as opinfo_definitions

# The min onnx opset version to test for
MIN_ONNX_OPSET_VERSION = 9
# The max onnx opset version to test for
MAX_ONNX_OPSET_VERSION = _constants.ONNX_MAX_OPSET

TESTED_OPSETS = range(MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION + 1)

LIMIT_SAMPLE_PER_OP = 10


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
    Tuple[OpInfo, torch.nn.Module, tuple, torch.dtype, Any]
]:
    """Produce samples of all operators to test."""

    op_db = itertools.chain(
        common_methods_invocations.op_db,
        opinfo_definitions.op_db,
    )
    for op_info in op_db:
        for dtype in TESTED_DTYPES:
            if op_info.name == "nn.functional.conv2d" or op_info.aten_name == "conv2d":
                # FIXME: Why do we not see nn.functional.conv2d?
                print(ValueError(op_info))
                exit()
            try:
                for sample in op_info.sample_inputs(
                    device="cpu", dtype=dtype, requires_grad=False
                )[:LIMIT_SAMPLE_PER_OP]:
                    model = SingleOpModel(op_info.op, sample.kwargs)
                    # Try to run it once. If it fails, skip it.
                    model(sample.input, *sample.args)
                    yield op_info, model, (
                        sample.input,
                        *sample.args,
                    ), dtype, sample
            except Exception:
                # Skip operators that don't support the dtype
                # E.g. "normal_kernel_cpu" not implemented for 'Bool'
                # Or Got unsupported ScalarType ComplexHalf
                pass


@dataclasses.dataclass
class OpTestResult:
    """Result of testing an operator."""

    opset: int
    dtype: torch.dtype
    operator: str
    aten_name: str
    exception: Optional[Exception]
    traceback: Optional[str]
    inputs: Tuple
    kwargs: Dict[str, Any]


class ResultCollection:
    def __init__(self) -> None:
        self.collection: Dict[Tuple[str, str, int], List[OpTestResult]] = {}

    def add(self, result: OpTestResult) -> None:
        key = (result.operator, str(result.dtype), result.opset)
        if key not in self.collection:
            self.collection[key] = []
        self.collection[key].append(result)

    def as_dict(self) -> List:
        """Convert the collection to a dict."""
        # Filter out the None values
        return [
            {
                "operator": key[0],
                "dtype": key[1],
                "opset": key[2],
                "exceptions": [
                    {
                        "type": type(result.exception).__name__,
                        "message": str(result.exception),
                        "inputs": repr(result.inputs),
                        "kwargs": repr(result.kwargs),
                        "traceback": result.traceback,
                    }
                    for result in value
                    if result.exception is not None
                ],
                "correct": [result.exception for result in value].count(None),
                "total": len(value),
            }
            for key, value in self.collection.items()
        ]


def check_single_op(
    op_info: OpInfo,
    model: torch.nn.Module,
    inputs: tuple,
    dtype: torch.dtype,
    opset_version: int,
    sample: Any,
) -> OpTestResult:
    # Export the model
    model_buffer = io.BytesIO()

    try:
        # TODO: Catch the warnings
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
            aten_name=op_info.aten_name,
            exception=e,
            traceback=traceback.format_exc(),
            inputs=inputs,
            kwargs=sample.kwargs,
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
            aten_name=op_info.aten_name,
            exception=e,
            traceback=traceback.format_exc(),
            inputs=inputs,
            kwargs=sample.kwargs,
        )

    return OpTestResult(
        opset=opset_version,
        dtype=dtype,
        operator=op_info.name,
        aten_name=op_info.aten_name,
        exception=None,
        traceback=None,
        inputs=inputs,
        kwargs=sample.kwargs,
    )


def test_op_consistency(opset_version: int) -> List[OpTestResult]:
    """Test that torch.onnx export produces the same results as aten."""
    results = []
    print("Producing samples...")
    all_samples = produce_op_sample()

    for i, (op_info, model, inputs, dtype, sample) in tqdm.tqdm(enumerate(all_samples)):
        result = check_single_op(op_info, model, inputs, dtype, opset_version, sample)
        results.append(result)

    return results


def main():
    collection = ResultCollection()

    for opset_version in TESTED_OPSETS:
        print(f"Testing opset {opset_version}")
        results = test_op_consistency(opset_version)
        for result in results:
            collection.add(result)
    # Save results to a json file
    with open("op_survey.json", "w") as f:
        json.dump(collection.as_dict(), f, indent=2)


if __name__ == "__main__":
    main()
