"""Utility functions for loading and processing datasets."""

import json
import os
from typing import Any

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.system_prompts import LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE, LCB_FORMATTING_WITHOUT_STARTER_CODE, LCB_SYSTEM_MESSAGE_GENERIC


def load_dataset(dataset_enum: TrainDataset.Math | TrainDataset.Code | TestDataset.Math | TestDataset.Code) -> list[dict[str, Any]]:
    """Load a dataset from a JSON file based on the dataset enum.

    This function takes a dataset enum value and loads the corresponding JSON file
    from the appropriate directory structure. The directory structure follows the pattern:
    {data_dir}/{category_dir}/{dataset_name}.json
    where:
    - data_dir is either 'train' or 'test'
    - category_dir is either 'math' or 'code'
    - dataset_name is the lowercase value of the enum

    Args:
        dataset_enum: An enum value from either TrainDataset or TestDataset classes,
                     specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset items.
            Each dictionary represents one item in the dataset with its associated fields.

    Raises:
        ValueError: If the dataset file cannot be found or contains invalid JSON.

    Examples:
        >>> # Load training AIME dataset
        >>> aime_data = load_dataset(TrainDataset.Math.AIME)
        >>> # Load test APPS dataset
        >>> apps_data = load_dataset(TestDataset.Code.APPS)
    """
    dataset_name = dataset_enum.value.lower()
    category_dir = dataset_enum.__class__.__name__.lower()

    # Determine if dataset is for training or testing
    if dataset_enum.__class__ in [TrainDataset.Math, TrainDataset.Code, TrainDataset.Web]:
        data_dir = "train"
    else:
        data_dir = "test"

    # Construct file path
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = "/scratch/m000123/context_manager_caroline/rllm/data"

    file_path = os.path.join(current_dir, data_dir, category_dir, f"{dataset_name}.json")

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}") from None
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}") from e


def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


if __name__ == "__main__":
    # Example usage
    aime_data = load_dataset(TrainDataset.Math.AIME)
    print(f"Loaded {len(aime_data)} AIME training problems")

    apps_data = load_dataset(TrainDataset.Code.APPS)
    print(f"Loaded {len(apps_data)} APPS test problems")

    lcb_data = load_dataset(TestDataset.Code.LIVECODEBENCH)
    print(f"Loaded {len(lcb_data)} Livecodebench test problems")

    code_contests_data = load_dataset(TestDataset.Code.CODE_CONTESTS)
    print(f"Loaded {len(code_contests_data)} Code Contests test problems")

    codeforces_data = load_dataset(TestDataset.Code.CODEFORCES)
    print(f"Loaded {len(codeforces_data)} Codeforces test problems")
