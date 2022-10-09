"""Survey torch.fx support models."""

import json
import os
import traceback
from typing import Any, List

import torch
import torch.fx
import tqdm
from torch.testing._internal.opinfo.core import OpInfo

import common


def check_single_op(
    op_info: OpInfo,
    model: torch.nn.Module,
    inputs: tuple,
    dtype: torch.dtype,
    sample: Any,
) -> common.OpTestResult:
    try:
        # Symbolic tracing frontend - captures the semantics of the module
        torch.fx.symbolic_trace(model)
    except Exception as e:
        # TODO: Test in place variants as well
        # FIXME: Doesn't work now because of
        # Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanation of what types of control flow can be traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors
        return common.OpTestResult(
            opset="torch_fx",
            dtype=dtype,
            operator=op_info.name,
            aten_name=op_info.aten_name,
            exception=e,
            traceback=traceback.format_exc(),
            inputs=inputs,
            kwargs=sample.kwargs,
        )

    return common.OpTestResult(
        opset="torch_fx",
        dtype=dtype,
        operator=op_info.name,
        aten_name=op_info.aten_name,
        exception=None,
        traceback=None,
        inputs=inputs,
        kwargs=sample.kwargs,
    )


def test_op_consistency(all_samples) -> List[common.OpTestResult]:
    """Test that torch.onnx export produces the same results as aten."""
    results = []

    for i, (op_info, model, inputs, dtype, sample) in (
        pbar := tqdm.tqdm(
            enumerate(all_samples),
            total=len(all_samples),
            desc="Testing torch.fx",
        )
    ):
        pbar.set_postfix({"dtype": dtype, "op": op_info.name})
        result = check_single_op(op_info, model, inputs, dtype, sample)
        results.append(result)

    return results


def main():
    collection = common.ResultCollection()

    print("Producing samples...")
    all_samples = list(common.produce_op_sample())

    results = test_op_consistency(all_samples)
    for result in results:
        collection.add(result)
    # Save results to a json file
    print("Saving results...")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    results_dict = {
        "torch_version": torch.__version__,
        "test_results": collection.as_dict(),
    }
    with open(os.path.join(out_dir, f"op_survey_torch_fx.json"), "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    main()
