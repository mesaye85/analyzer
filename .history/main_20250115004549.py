import os
import re
import warnings
import pandas as pd
from pathlib import Path
from urllib3.exceptions import NotOpenSSLWarning
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import nltk

# Download NLTK resources
nltk.download("punkt")




# ===============================
# Utility Functions
# ===============================

def extract_text_from_txt(file_path):
    """Extract text content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def ensure_directory(path):
    """Ensure the directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ===============================
# IOC Extraction
# ===============================

def extract_iocs(text):
    """Extract IOCs using regex patterns."""
    patterns = {
        "ips": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ipv6": r"\b([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b",
        "urls": r"https?://[-\w.%[\da-fA-F]{2}]+",
        "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "hashes": r"\b[a-fA-F0-9]{32,64}\b",
        "domains": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
    }

    iocs = {key: re.findall(pattern, text) for key, pattern in patterns.items()}
    return iocs


def process_all_files(directory):
    """Process all text files in a directory and extract IOCs."""
    results = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            text = extract_text_from_txt(file_path)
            iocs = extract_iocs(text)
            results.append({"file": filename, "iocs": iocs})
    return results


def save_to_csv(data, output_file):
    """Save extracted IOCs to a CSV file."""
    rows = [
        {"file": item["file"], "ioc_type": ioc_type, "ioc": ioc}
        for item in data
        for ioc_type, iocs in item["iocs"].items()
        for ioc in iocs
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


# ===============================
# Summarization
# ===============================

def split_text_into_chunks(text, max_chunk_size=512):
    """Split long text into smaller chunks."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_text(text, model_name="t5-small"):
    """Summarize text using a pre-trained transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_long_text(text, model_name="t5-small"):
    """Summarize long text by chunking."""
    summaries = [summarize_text(chunk, model_name) for chunk in split_text_into_chunks(text)]
    return " ".join(summaries)


def clean_summary(summary):
    """Clean and post-process the generated summary."""
    sentences = summary.split(". ")
    unique_sentences = list(dict.fromkeys(sentences))
    cleaned_summary = ". ".join(unique_sentences)

    # Remove URLs, special characters, or noise
    cleaned_summary = re.sub(r"https?://\S+|www\.\S+", "", cleaned_summary)
    cleaned_summary = re.sub(r"[^A-Za-z0-9\s.,]", "", cleaned_summary)

    return cleaned_summary.strip()


def process_files_with_summary(dir_path, model_name="t5-small", output_file="summaries.txt"):
    """Summarize files incrementally and save summaries to disk."""
    ensure_directory(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as output:
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                text = extract_text_from_txt(file_path)
                summary = summarize_long_text(text, model_name)
                cleaned_summary = clean_summary(summary)
                output.write(f"Summary for {filename}:\n{cleaned_summary}\n\n")
                print(f"Processed {filename}")


# ===============================
# Neural Network Model
# ===============================

def build_model(vocab_size=5000, embedding_dim=100, input_length=200):
    """Build and compile the neural network model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(data, labels):
    """Train the neural network model."""
    model = build_model()
    model.fit(data, labels, epochs=10, validation_split=0.2, batch_size=32)
    model.save("models/ioc_detection")
    print("Model saved to 'models/ioc_detection'")


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    raw_reports_dir = "raw_reports"
    ensure_directory("output")

    # Extract IOCs
    processed_data = process_all_files(raw_reports_dir)
    save_to_csv(processed_data, "output/extracted_iocs.csv")

    # Summarize Text
    process_files_with_summary(raw_reports_dir, output_file="output/summaries.txt")
