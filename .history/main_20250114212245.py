import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import spacy
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy for sentence tokenization
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])
except OSError:
    logger.info("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])

# ===============================
# Utility Functions
# ===============================

def validate_file(file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    if path.suffix.lower() != '.txt':
        raise ValueError(f"Invalid file type: {path.suffix}")
    if path.stat().st_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File too large: {file_path}")

    return True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    try:
        validate_file(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        logger.error(f"Failed to decode file: {file_path}")
        raise

# ===============================
# IOC Extraction
# ===============================

def extract_iocs(text: str) -> Dict[str, List[str]]:
    patterns = {
        "ips": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "domains": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
        "urls": r"https?://[-\w.%[\da-fA-F]{2}]+",
        "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    }

    iocs = {}
    for ioc_type, pattern in patterns.items():
        iocs[ioc_type] = list(set(re.findall(pattern, text)))

    return iocs

# ===============================
# Summarization
# ===============================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def summarize_long_text(text: str, model_name: str = "t5-small") -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ===============================
# File Processing
# ===============================

def process_files(dir_path: str, model_name: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ioc_results = []
    summary_results = []

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        try:
            text = extract_text_from_txt(file_path)
            logger.info(f"Processing file: {filename}")

            # Extract IOCs
            iocs = extract_iocs(text)
            ioc_results.append({"file": filename, "iocs": iocs})

            # Generate Summary
            summary = summarize_long_text(text, model_name)
            summary_results.append({"file": filename, "summary": summary})
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    # Save IOCs
    if ioc_results:
        ioc_df = pd.DataFrame([
            {"file": item["file"], "ioc_type": ioc_type, "ioc": ioc}
            for item in ioc_results
            for ioc_type, iocs in item["iocs"].items()
            for ioc in iocs
        ])
        ioc_df.to_csv(os.path.join(output_dir, "extracted_iocs.csv"), index=False)
        logger.info("Saved extracted IOCs to output/extracted_iocs.csv")

    # Save Summaries
    with open(os.path.join(output_dir, "summaries.txt"), "w", encoding="utf-8") as f:
        for item in summary_results:
            f.write(f"Summary for {item['file']}\n{item['summary']}\n\n")
        logger.info("Saved summaries to output/summaries.txt")

# ===============================
# Main Execution
# ===============================

def main():
    try:
        raw_reports_dir = "raw_reports"
        output_dir = "output"

        if not os.path.exists(raw_reports_dir):
            logger.error(f"Input directory '{raw_reports_dir}' does not exist.")
            return

        process_files(raw_reports_dir, model_name="t5-small", output_dir=output_dir)
        logger.info("Analysis completed successfully.")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")

if __name__ == "__main__":
    main()
