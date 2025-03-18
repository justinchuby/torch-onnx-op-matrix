"""Survey torch.onnx.export supported models."""

import argparse
import json
import os
import traceback
from typing import Any, List
import warnings

import torch
import tqdm
import onnx
import onnxscript
from torch.testing._internal.opinfo.core import OpInfo
import torch.onnx._internal.diagnostics.infra.context

import common


def check_single_op(
    op_info: OpInfo,
    model: torch.nn.Module,
    inputs: tuple,
    dtype: torch.dtype,
    sample: Any,
) -> common.OpTestResult:
    torch._dynamo.reset()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = torch.onnx.export(
                model,
                inputs,
                dynamo=True,
                verbose=False,
            )
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

    # Check the model with ONNX
    onnx_model = output.model_proto
    try:
        onnx.checker.check_model(onnx_model, full_check=True)  # type: ignore
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
            desc="Testing torch.onnx.export dynamo",
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
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    results_dict = {
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxscript_version": onnxscript.__version__,
        "test_results": collection.as_dict(),
    }
    # Save results to a json file
    out_path = os.path.join(out_dir, "onnx_dynamo_op_survey.json")
    print(f"Saving results to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug run.",
    )
    main(parser.parse_args())
