"""Test consistency between torch.onnx exported operators and aten operators."""

import itertools
import dataclasses
from typing import Any, Dict, Iterator, List, Optional, AbstractSet, Tuple
import warnings

import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo import definitions as opinfo_definitions

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


def produce_op_sample(skip_ops: AbstractSet[str] | None = None) -> Iterator[
    Tuple[OpInfo, torch.nn.Module, tuple, torch.dtype, Any]
]:
    """Produce samples of all operators to test."""
    skip_ops = skip_ops or set()

    op_db = itertools.chain(
        common_methods_invocations.op_db,
        opinfo_definitions.op_db,
    )
    for op_info in op_db:
        if op_info.name in skip_ops:
            # For some reason we have Floating point exception(core dumped) in github actions
            continue
        for dtype in TESTED_DTYPES:
            try:
                for i, sample in enumerate(
                    op_info.sample_inputs(
                        device="cpu", dtype=dtype, requires_grad=False
                    )
                ):
                    if i >= LIMIT_SAMPLE_PER_OP:
                        break
                    model = SingleOpModel(op_info.op, sample.kwargs)
                    # Run the model to make sure PyTorch can run it first
                    model(sample.input, *sample.args)
                    yield op_info, model, (
                        sample.input,
                        *sample.args,
                    ), dtype, sample
            except Exception as e:
                # Skip operators that don't support the dtype
                # E.g. "normal_kernel_cpu" not implemented for 'Bool'
                # Or Got unsupported ScalarType ComplexHalf
                warnings.warn(f"!!!Skipping {op_info.name} for {dtype}: {e}")


@dataclasses.dataclass
class OpTestResult:
    """Result of testing an operator."""

    opset: int | str
    dtype: torch.dtype
    operator: str
    aten_name: str
    exception: Optional[Exception]
    traceback: Optional[str]
    inputs: Tuple
    kwargs: Dict[str, Any]


class ResultCollection:
    def __init__(self) -> None:
        self.collection: Dict[Tuple[str, str, int | str, str], List[OpTestResult]] = {}

    def add(self, result: OpTestResult) -> None:
        key = (result.operator, str(result.dtype), result.opset, result.aten_name)
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
                "aten_name": key[3],
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
