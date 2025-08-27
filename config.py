# config.py

# --- API and Model Configuration ---
# IMPORTANT: It's best practice to load secrets from environment variables
# rather than hardcoding them in the file.
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = 'gemini-1.5-flash-latest' # Or any other model you prefer

# --- Dataset and Hub Configuration ---
SOURCE_DATASET_NAME = "TheFinAI/Fino1_Reasoning_Path_FinQA"
TARGET_HUB_REPO_NAME = "YoussefHosni/Fino1_Reasoning_Path_FinQA_Arabic" # Your HF Hub repo
PUSH_TO_HUB = True  # Set to False for local-only testing

# --- Directory Configuration ---
OUTPUT_DIR = "./translated_data"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
FINAL_FILENAME = "finqa_arabic_complete.jsonl"

# --- Processing Configuration ---
# Set START_INDEX to 0 to start from the beginning.
# If the script is interrupted at index 1567, you can set START_INDEX = 1500 (the start of that batch)
# to resume without re-translating completed batches.
START_INDEX = 4000
BATCH_SIZE = 100  # Process 100 examples and then save a checkpoint
NUM_SAMPLES_TO_PROCESS = None  # Use None to process the entire dataset, or a number (e.g., 500) for a subset

# --- Translation Logic Configuration ---
# Define which columns/fields in the dataset need to be translated.
FIELDS_TO_TRANSLATE = ['Open-ended Verifiable Question', 'Ground-True Answer', 'Complex_CoT', 'Response']

# --- Retry Logic ---
# How many times to retry the API call if it fails with a transient error.
API_RETRIES = 3
# How many seconds to wait between retries.
API_RETRY_DELAY = 5