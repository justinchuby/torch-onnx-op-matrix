"""Test consistency between torch.onnx exported operators and aten operators."""

import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, AbstractSet, Tuple

import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.opinfo.core import OpInfo

LIMIT_SAMPLE_PER_OP = 10


TESTED_DTYPES = (
    torch.bool,
    # torch.uint8,
    # torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    # Floating types
    torch.float16,
    torch.float32,
    torch.float64,
    # torch.bfloat16,
    # Complex types
    # torch.complex32,
    torch.complex64,
    # torch.complex128,
)


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


class SingleOpModelZeroInputs(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self):
        return self.operator(**self.kwargs)


class SingleOpModelOneInput(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, x):
        return self.operator(x, **self.kwargs)


class SingleOpModelTwoInputs(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, x, y):
        return self.operator(x, y, **self.kwargs)


class SingleOpModelThreeInputs(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, x, y, z):
        return self.operator(x, y, z, **self.kwargs)


class SingleOpModelFourInputs(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, x, y, z, a):
        return self.operator(x, y, z, a, **self.kwargs)


class SingleOpModelFiveInputs(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, x, y, z, a, b):
        return self.operator(x, y, z, a, b, **self.kwargs)


def produce_op_sample(
    skip_ops: AbstractSet[str] | None = None, target: str | None = None
) -> Iterator[Tuple[OpInfo, torch.nn.Module, tuple, torch.dtype, Any]]:
    """Produce samples of all operators to test."""
    skip_ops = skip_ops or set()

    # opinfo_definitions.op_db is part of common_methods_invocations.op_db
    op_db = common_methods_invocations.op_db
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

                    inputs = (
                        sample.input,
                        *sample.args,
                    )
                    model: torch.nn.Module
                    if target == "fx":
                        if len(inputs) == 0:
                            model = SingleOpModelZeroInputs(op_info.op, sample.kwargs)
                        elif len(inputs) == 1:
                            model = SingleOpModelOneInput(op_info.op, sample.kwargs)
                        elif len(inputs) == 2:
                            model = SingleOpModelTwoInputs(op_info.op, sample.kwargs)
                        elif len(inputs) == 3:
                            model = SingleOpModelThreeInputs(op_info.op, sample.kwargs)
                        elif len(inputs) == 4:
                            model = SingleOpModelFourInputs(op_info.op, sample.kwargs)
                        elif len(inputs) == 5:
                            model = SingleOpModelFiveInputs(op_info.op, sample.kwargs)
                        else:
                            raise ValueError(
                                "torch_fx model requiring >5 inputs is ignored"
                            )
                    else:
                        model = SingleOpModel(op_info.op, sample.kwargs)
                    # Run the model to make sure PyTorch can run it first
                    model(*inputs)
                    yield op_info, model, inputs, dtype, sample
            except Exception as e:
                # Skip operators that don't support the dtype
                # E.g. "normal_kernel_cpu" not implemented for 'Bool'
                # Or Got unsupported ScalarType ComplexHalf
                logging.warn(f"!!!Skipping {op_info.name} for {dtype}: {e}")


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
