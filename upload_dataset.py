# upload_dataset.py

import argparse
import logging
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Import settings from the config file to use as defaults
import config

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_repo_name_from_env():
    """
    Determines the Hugging Face repository name based on environment variables.

    Environment Variables:
        UPLOAD_TARGET (str): "personal" or "organization". Defaults to "personal".
        PERSONAL_HUB_REPO_PATH (str): The path for personal uploads (e.g., "username/repo").
        ORG_HUB_REPO_PATH (str): The path for organization uploads (e.g., "org-name/repo").

    Returns:
        str or None: The determined repository name or None if not configured.
    """
    upload_target = os.environ.get("UPLOAD_TARGET", "personal").lower()
    repo_name = None

    if upload_target == "organization":
        repo_name = os.environ.get("ORG_HUB_REPO_PATH")
        if not repo_name:
            logging.warning("UPLOAD_TARGET is 'organization', but ORG_HUB_REPO_PATH is not set.")
    elif upload_target == "personal":
        repo_name = os.environ.get("PERSONAL_HUB_REPO_PATH")
        if not repo_name:
            logging.warning("UPLOAD_TARGET is 'personal', but PERSONAL_HUB_REPO_PATH is not set.")
    else:
        logging.error(f"Invalid UPLOAD_TARGET: '{upload_target}'. Please use 'personal' or 'organization'.")

    return repo_name


def upload_to_hub(file_path, repo_name):
    """
    Loads a dataset from a .jsonl file, creates a DatasetDict,
    and pushes it to the Hugging Face Hub.

    Args:
        file_path (str): The local path to the .jsonl file.
        repo_name (str): The name of the repository on the Hugging Face Hub.
    """
    # 1. Get the Hugging Face token from environment variables
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logging.error("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        return

    # 2. Validate that a repository name is provided
    if not repo_name:
        logging.error("Repository name is not provided. Please set the environment variables or use the --repo-name flag.")
        return

    # 3. Validate that the input file exists
    if not os.path.exists(file_path):
        logging.error(f"Input file not found at: {file_path}")
        return

    logging.info(f"Preparing to upload '{file_path}' to '{repo_name}'.")

    try:
        # 4. Load the dataset from the JSONL file
        logging.info("Loading dataset from JSONL file...")
        dataset = Dataset.from_json(file_path)

        # 5. Create a standard DatasetDict format
        final_dataset_dict = DatasetDict({"train": dataset})
        logging.info("Dataset loaded and formatted successfully:")
        print(final_dataset_dict)

        # 6. Push to the Hub, providing the token for authentication
        logging.info(f"Pushing to Hugging Face Hub. This may take a while...")
        final_dataset_dict.push_to_hub(
            repo_id=repo_name,
            token=hf_token  # Authenticate the request
        )

        logging.info("=" * 30)
        logging.info("✅ Successfully pushed dataset to the Hub!")
        logging.info(f"View your dataset at: https://huggingface.co/datasets/{repo_name}")
        logging.info("=" * 30)

    except Exception as e:
        logging.error(f"An error occurred during the upload process: {e}")
        logging.error("Please ensure the file is a valid JSONL, you are logged into Hugging Face, and the repo name is correct.")


if __name__ == "__main__":
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Upload a translated dataset to the Hugging Face Hub using environment variables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_file_path = os.path.join(config.OUTPUT_DIR, config.FINAL_FILENAME)

    parser.add_argument(
        "--file-path",
        type=str,
        default=default_file_path,
        help="Path to the final .jsonl file to upload."
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Override the Hugging Face Hub repo name. If not set, it's determined by environment variables."
    )

    args = parser.parse_args()

    final_repo_name = args.repo_name or get_repo_name_from_env()

    # --- Run the Upload ---
    upload_to_hub(args.file_path, final_repo_name)