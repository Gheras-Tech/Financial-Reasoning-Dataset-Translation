# Financial Reasoning Dataset Translation using Gemini

This project provides a modular and resumable Python script to translate [financial reasoning datasets](https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA) from English to Arabic using the Google Gemini API. The workflow is split into two main steps:
1. **Translate:** Process a dataset in batches, save checkpoints, and create a final consolidated `.jsonl` file.
2. **Upload:** Push the final `.jsonl` file to the Hugging Face Hub with flexible repository targeting.

## Features

- **Modular:** Translation and uploading are separate, independent scripts.
- **Resumable:** Translates in batches and creates checkpoints. If the translation script is interrupted, it can resume from the last completed batch.
- **Robust:** Includes automatic retry logic for transient API errors.
- **Flexible Upload:** Support for both personal and organization repositories with environment-based configuration.
- **High-Quality Translation:** Optimized prompts for financial and mathematical terminology in Modern Standard Arabic (MSA).
- **Progress Tracking:** Detailed logging and progress bars for monitoring translation progress.

## Project Structure

```
Financial_Reasoning_Dataset_Translation/
├── translate_script.py      # Translates the dataset and saves it locally
├── upload_dataset.py        # Pushes the final dataset file to the Hub
├── config.py                # All your settings and configurations
├── requirements.txt         # All the necessary Python packages
├── .env                     # Environment variables (create this file)
├── translated_data/         # Output directory for translated data
│   ├── checkpoints/         # Batch checkpoint files
│   └── finqa_arabic_complete.jsonl  # Final consolidated dataset
└── README.md                # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure the Script

Open the `config.py` file and review the settings:
- `SOURCE_DATASET_NAME`: The source dataset to translate
- `START_INDEX`: Where to start translation (useful for resuming)
- `BATCH_SIZE`: Number of examples to process per batch
- `FIELDS_TO_TRANSLATE`: Which dataset columns to translate

### 3. Set Up Authentication

Create a `.env` file in the root directory with your API keys and tokens:

```bash
# .env
GEMINI_API_KEY=your_google_gemini_api_key
HF_TOKEN=your_hugging_face_write_token

# Upload Configuration
UPLOAD_TARGET="organization"  # or "personal"
PERSONAL_HUB_REPO_PATH="your-username/repo-name"
ORG_HUB_REPO_PATH="org-name/repo-name"
```

**Important Notes:**
- **Gemini API Key:** Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Hugging Face Token:** Get from [HF Settings](https://huggingface.co/settings/tokens) with "Write" permissions
- **Repository Paths:** Use format "username/repo-name" or "org-name/repo-name"

## How to Run

### Step 1: Translate the Dataset

This script reads your `GEMINI_API_KEY` from the `.env` file to authenticate with the Gemini API.

```bash
python translate_script.py
```

**Translation Process:**
- Loads the source dataset from Hugging Face Hub
- Processes examples in configurable batches
- Saves checkpoints after each batch (enables resuming)
- Translates specified fields to Modern Standard Arabic
- Creates final consolidated `.jsonl` file

**Resuming Translation:**
If interrupted, modify `START_INDEX` in `config.py` to resume from a specific batch.

### Step 2: Upload the Translated Dataset

This script reads your `HF_TOKEN` from the `.env` file to authenticate with the Hugging Face Hub.

```bash
# Upload using environment variables (recommended)
python upload_dataset.py

# Or specify custom file path and repository
python upload_dataset.py --file-path ./translated_data/finqa_arabic_complete.jsonl --repo-name "your-username/custom-repo"
```

**Upload Options:**
- **Environment-based:** Uses `UPLOAD_TARGET`, `PERSONAL_HUB_REPO_PATH`, and `ORG_HUB_REPO_PATH`
- **Command-line override:** Use `--repo-name` to override environment settings
- **Flexible targeting:** Support for both personal and organization repositories

## Configuration Details

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | `AIzaSy...` |
| `HF_TOKEN` | Hugging Face write token | `hf_...` |
| `UPLOAD_TARGET` | Upload destination | `"personal"` or `"organization"` |
| `PERSONAL_HUB_REPO_PATH` | Personal repo path | `"username/repo-name"` |
| `ORG_HUB_REPO_PATH` | Organization repo path | `"org-name/repo-name"` |

### Config.py Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `SOURCE_DATASET_NAME` | Source dataset | `"TheFinAI/Fino1_Reasoning_Path_FinQA"` |
| `START_INDEX` | Resume point | `4000` |
| `BATCH_SIZE` | Batch size | `100` |
| `FIELDS_TO_TRANSLATE` | Fields to translate | `['Open-ended Verifiable Question', 'Ground-True Answer', 'Complex_CoT', 'Response']` |
| `API_RETRIES` | Retry attempts | `3` |
| `API_RETRY_DELAY` | Retry delay (seconds) | `5` |

## Troubleshooting

### Common Issues

1. **401 Unauthorized Error:**
   - Ensure your `HF_TOKEN` is valid and has write permissions
   - Check if you have access to the target organization
   - Verify the token hasn't expired

2. **Translation Interruption:**
   - Check `START_INDEX` in `config.py` to resume from the correct batch
   - Verify checkpoint files exist in `translated_data/checkpoints/`

3. **API Rate Limits:**
   - The script includes automatic retry logic
   - Adjust `API_RETRY_DELAY` in `config.py` if needed

### Getting Help

- Check the logs for detailed error messages
- Verify all environment variables are set correctly
- Ensure you have the required permissions for the target repository

## Output

The translation process creates:
- **Checkpoint files:** `translated_data/checkpoints/batch_X-Y.jsonl`
- **Final dataset:** `translated_data/finqa_arabic_complete.jsonl`
- **Hugging Face dataset:** Available at the specified repository URL

The final dataset maintains the original structure with Arabic translations of the specified fields.
