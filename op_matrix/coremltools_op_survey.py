"""Survey coremltools converter supported models."""

import json
import os
import traceback
from typing import Any, List

import coremltools as ct
import torch
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
        trace = torch.jit.trace(model, inputs)

        # Convert the model
        ct.convert(
            trace,
            inputs=[
                ct.TensorType(name=f"input_{i}", shape=input_.shape)
                for i, input_ in enumerate(inputs)
            ],
        )
    except Exception as e:
        # TODO: Test in place variants as well
        return common.OpTestResult(
            opset="coremltools",
            dtype=dtype,
            operator=op_info.name,
            aten_name=op_info.aten_name,
            exception=e,
            traceback=traceback.format_exc(),
            inputs=inputs,
            kwargs=sample.kwargs,
        )

    return common.OpTestResult(
        opset="coremltools",
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
            desc="Testing coremltools",
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
        "coremltools_version": ct.__version__,
        "test_results": collection.as_dict(),
    }
    with open(os.path.join(out_dir, "op_survey_coremltools.json"), "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    main()
