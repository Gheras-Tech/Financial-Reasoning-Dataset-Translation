# translate_script.py (Updated: Upload logic removed)

import os
import json
import time
import logging
import google.generativeai as genai
from google.api_core import exceptions
from datasets import load_dataset
from tqdm.auto import tqdm

# Import all settings from the config file
import config

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_model(api_key, model_name):
    """Configures the API and initializes the generative model."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        logging.info(f"Successfully initialized model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        raise

def translate_text(text, model, retries, delay):
    """Translates a single string using the Gemini model with retry logic."""
    if not text or not isinstance(text, str) or not text.strip():
        return text

    prompt = f"""Translate the following English text into high-quality Modern Standard Arabic (MSA) with a focus on clarity, precise financial and mathematical terminology, and natural fluency for educational or reasoning-based datasets. Maintain all numbers, special characters, and equations exactly as in the original.

    Original English Text:
    {text}

    Translated Arabic Text:"""

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except (exceptions.DeadlineExceeded, exceptions.ServiceUnavailable, exceptions.ResourceExhausted) as e:
            logging.warning(f"Attempt {attempt + 1}/{retries} failed for '{text[:50]}...': {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Max retries reached for '{text[:50]}...'. Returning error placeholder.")
                return f"[TRANSLATION_ERROR_GEMINI_RETRY] {text}"
        except Exception as e:
            logging.error(f"An unexpected error occurred translating '{text[:50]}...': {e}")
            return f"[TRANSLATION_ERROR_GEMINI_UNEXPECTED] {text}"

def translate_example(example, model, fields_to_translate):
    """Translates all specified fields of a dataset example."""
    translated_example = example.copy()
    for field in fields_to_translate:
        if field in example and example[field]:
            original_text = str(example[field])
            translated_example[field] = translate_text(original_text, model, config.API_RETRIES, config.API_RETRY_DELAY)
    return translated_example

def process_batches(dataset, model, start_index, end_index, batch_size):
    """Iterates through the dataset in batches, translates, and saves checkpoints."""
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        logging.info(f"Created checkpoint directory: {config.CHECKPOINT_DIR}")

    for i in range(start_index, end_index, batch_size):
        batch_start_index = i
        batch_end_index = min(i + batch_size, end_index)
        
        logging.info(f"--- Processing Batch: Indices {batch_start_index} to {batch_end_index-1} ---")
        
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"batch_{batch_start_index}-{batch_end_index-1}.jsonl")
        if os.path.exists(checkpoint_path):
            logging.info(f"Checkpoint already exists, skipping: {checkpoint_path}")
            continue

        batch_dataset = dataset.select(range(batch_start_index, batch_end_index))
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f_out:
            for example in tqdm(batch_dataset, desc=f"Translating Batch {batch_start_index}-{batch_end_index-1}"):
                translated_ex = translate_example(example, model, config.FIELDS_TO_TRANSLATE)
                f_out.write(json.dumps(translated_ex, ensure_ascii=False) + "\n")
        
        logging.info(f"Batch complete. Checkpoint saved to: {checkpoint_path}")

def consolidate_checkpoints():
    """Combines all batch checkpoint files into a single final file."""
    logging.info("Combining all checkpoint files...")
    final_output_path = os.path.join(config.OUTPUT_DIR, config.FINAL_FILENAME)
    
    try:
        checkpoint_files = sorted(
            [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.jsonl')],
            key=lambda x: int(x.split('_')[1].split('-')[0])
        )
    except (FileNotFoundError, IndexError):
        logging.warning("No checkpoint files found to consolidate.")
        return None

    total_lines = 0
    with open(final_output_path, 'w', encoding='utf-8') as f_final:
        for filename in checkpoint_files:
            with open(os.path.join(config.CHECKPOINT_DIR, filename), 'r', encoding='utf-8') as f_batch:
                for line in f_batch:
                    f_final.write(line)
                    total_lines += 1
    
    if total_lines > 0:
        logging.info(f"✅ Successfully combined {len(checkpoint_files)} batches into '{final_output_path}' ({total_lines} examples).")
        return final_output_path
    else:
        logging.warning("Consolidation resulted in an empty file.")
        return None

def main():
    """Main function to orchestrate the translation pipeline."""
    logging.info("--- Starting Dataset Translation Pipeline ---")
    
    model = initialize_model(config.GEMINI_API_KEY, config.MODEL_NAME)
    
    logging.info(f"Loading source dataset: {config.SOURCE_DATASET_NAME}")
    dataset = load_dataset(config.SOURCE_DATASET_NAME, split="train")
    
    total_examples = len(dataset)
    end_index = total_examples
    if config.NUM_SAMPLES_TO_PROCESS is not None:
        end_index = min(config.START_INDEX + config.NUM_SAMPLES_TO_PROCESS, total_examples)
    
    logging.info(f"Total examples in dataset: {total_examples}")
    logging.info(f"Processing from index {config.START_INDEX} to {end_index-1}")
    
    process_batches(dataset, model, config.START_INDEX, end_index, config.BATCH_SIZE)
    
    consolidate_checkpoints()
        
    logging.info("--- Translation and Consolidation Finished ---")
    logging.info(f"Final file is ready at: {os.path.join(config.OUTPUT_DIR, config.FINAL_FILENAME)}")
    logging.info("You can now run 'upload_dataset.py' to push it to the Hub.")

if __name__ == "__main__":
    main()