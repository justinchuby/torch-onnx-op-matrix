from collections import defaultdict
import json
import os
import argparse
import random


def process_data_for_visualization(
    data: dict, sample_exceptions: int | None = None
) -> list[dict]:
    results = defaultdict(dict)
    for test_result in data["test_results"]:
        operator = test_result["operator"]
        dtype = test_result["dtype"].split("torch.")[1]
        correct = test_result["correct"]
        total = test_result["total"]
        aten_name = test_result["aten_name"]
        exceptions = test_result["exceptions"]
        results[operator][dtype] = {
            "correct_count": correct,
            "total_count": total,
            "aten_name": aten_name,
            "exceptions": random.sample(exceptions, sample_exceptions)
            if sample_exceptions and len(exceptions) >= sample_exceptions
            else exceptions,
        }

    return [{"operator": operator, **results} for operator, results in results.items()]


def file_name_sort_key(file_name: str) -> tuple[int, str]:
    # Remove the file extension
    file_name = file_name.split(".")[0]
    parts = file_name.split("_")
    # Try to convert the last part to an int
    try:
        opset = int(parts[-1])
        s = ""
    except ValueError:
        # Set to a big number so that it is sorted last
        opset = 1000
        s = parts[-1]
    return opset, s


def main(args):
    results = []
    for file in sorted(os.listdir(args.input_dir), key=file_name_sort_key):
        if file.endswith(".json"):
            with open(os.path.join(args.input_dir, file), "r") as f:
                data = json.load(f)
                processed_data = process_data_for_visualization(
                    data, args.sample_exceptions
                )
                result = {
                    "file": file,
                    "torch_version": data["torch_version"],
                    "onnx_version": data.get("onnx_version"),
                    "opset": data["test_results"][0]["opset"],
                    "test_results": processed_data,
                }
                results.append(result)

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample_exceptions", type=int, default=2)
    args = parser.parse_args()
    main(args)
