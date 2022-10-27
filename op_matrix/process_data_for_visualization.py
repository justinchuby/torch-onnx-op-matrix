from collections import defaultdict
import json
import os
import argparse


def process_data_for_visualization(data: dict) -> list[dict]:
    results = defaultdict(dict)
    for test_result in data["test_results"]:
        operator = test_result["operator"]
        dtype = test_result["dtype"].split("torch.")[1]
        correct = test_result["correct"]
        total = test_result["total"]
        results[operator][dtype] = {"correct_count": correct, "total_count": total}

    return [{"operator": operator, **results} for operator, results in results.items()]


def main(args):
    with open(args.input, "r") as f:
        data = json.load(f)

    results = process_data_for_visualization(data)

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)
