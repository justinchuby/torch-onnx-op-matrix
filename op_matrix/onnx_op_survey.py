"""Survey torch.onnx supported models."""

import argparse
import json
import io
import logging
import multiprocessing
import os
import traceback
from typing import Any, List

import onnx
import torch
import tqdm
from torch.onnx import _constants
from torch.testing._internal.opinfo.core import OpInfo

import common


FLOATING_POINT_EXCEPTION_OPS = frozenset(
    [
        "nn.functional.pixel_unshuffle",
        "nn.functional.pixel_shuffle",
        "take",
    ]
)


def check_single_op(
    op_info: OpInfo,
    model: torch.nn.Module,
    inputs: tuple,
    dtype: torch.dtype,
    opset_version: int,
    sample: Any,
) -> common.OpTestResult:
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
        return common.OpTestResult(
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
        onnx.checker.check_model(onnx_model, full_check=True)  # type: ignore
    except onnx.checker.ValidationError as e:  # type: ignore
        return common.OpTestResult(
            opset=opset_version,
            dtype=dtype,
            operator=op_info.name,
            aten_name=op_info.aten_name,
            exception=e,
            traceback=traceback.format_exc(),
            inputs=inputs,
            kwargs=sample.kwargs,
        )

    return common.OpTestResult(
        opset=opset_version,
        dtype=dtype,
        operator=op_info.name,
        aten_name=op_info.aten_name,
        exception=None,
        traceback=None,
        inputs=inputs,
        kwargs=sample.kwargs,
    )


def test_op_consistency(opset_version: int, all_samples) -> List[common.OpTestResult]:
    """Test that torch.onnx export produces the same results as aten."""
    results = []

    for i, (op_info, model, inputs, dtype, sample) in (
        pbar := tqdm.tqdm(
            enumerate(all_samples),
            total=len(all_samples),
            desc=f"Testing opset {opset_version}",
        )
    ):
        pbar.set_postfix({"dtype": dtype, "op": op_info.name})
        result = check_single_op(op_info, model, inputs, dtype, opset_version, sample)
        results.append(result)

    return results


def run_one_opset(opset_version: int) -> None:
    collection = common.ResultCollection()

    print("Producing samples...")
    all_samples = list(common.produce_op_sample(FLOATING_POINT_EXCEPTION_OPS))

    print(f"Testing opset {opset_version}")
    results = test_op_consistency(opset_version, all_samples)
    for result in results:
        collection.add(result)
    # Save results to a json file
    print("Saving results...")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    results_dict = {
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "test_results": collection.as_dict(),
    }
    with open(os.path.join(out_dir, f"op_survey_opset_{opset_version}.json"), "w") as f:
        json.dump(results_dict, f, indent=2)


def main(args):
    opset_version = args.opset
    jobs = args.jobs

    if opset_version == -1:
        # Test all opsets
        opset_versions = range(9, _constants.ONNX_MAX_OPSET + 1)
    else:
        opset_versions = [opset_version]

    with multiprocessing.Pool(jobs) as pool:
        pool.map(run_one_opset, opset_versions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opset",
        type=int,
        default=_constants.ONNX_MAX_OPSET,
        help="The opset version to test.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="The number of processes to use for testing.",
    )
    main(parser.parse_args())
