"""Survey torch.fx support models."""

import argparse
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
        torch.onnx.dynamo_export(model, *inputs, **sample.kwargs)
    except Exception as e:
        return common.OpTestResult(
            opset="onnx_dynamo",
            dtype=dtype,
            operator=op_info.name,
            aten_name=op_info.aten_name,
            exception=e,
            traceback=traceback.format_exc(),
            inputs=inputs,
            kwargs=sample.kwargs,
        )

    return common.OpTestResult(
        opset="onnx_dynamo",
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
            desc="Testing torch.onnx.dynamo_export",
        )
    ):
        pbar.set_postfix({"dtype": dtype, "op": op_info.name})
        result = check_single_op(op_info, model, inputs, dtype, sample)
        results.append(result)

    return results


def main(args):
    debug = args.debug

    collection = common.ResultCollection()

    print("Producing samples...")
    all_samples = list(common.produce_op_sample(target="onnx_dynamo"))

    if debug:
        print("Debug mode, only testing a subset of samples")
        all_samples = all_samples[:10]

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
    with open(os.path.join(out_dir, f"op_survey_fx.json"), "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug run.",
    )
    main(parser.parse_args())
