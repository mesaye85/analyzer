import os
import re
import pandas as pd


def extract_text_from_txt(file_path):
    """Read text from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_iocs(text):
    """Extract IOCs using regex patterns."""
    ip_regex = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    ipv6_regex = re.compile(r"\b([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b")
    url_regex = re.compile(r"https?://[-\w.%[\da-fA-F]{2}]+")
    email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    hash_regex = re.compile(r"\b[a-fA-F0-9]{32,64}\b")  # MD5, SHA256
    domain_regex = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")

    return {
        "ips": ip_regex.findall(text) + ipv6_regex.findall(text),
        "urls": url_regex.findall(text),
        "emails": email_regex.findall(text),
        "hashes": hash_regex.findall(text),
        "domains": domain_regex.findall(text),
    }


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
    rows = []
    for item in data:
        for ioc_type, iocs in item["iocs"].items():
            for ioc in iocs:
                rows.append({"file": item["file"], "ioc_type": ioc_type, "ioc": ioc})
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    # Define the directory containing text files
    directory = "data/raw_reports"

    # Process all files and extract IOCs
    processed_data = process_all_files(directory)

    # Save the results to a CSV file
    save_to_csv(processed_data, "extracted_iocs.csv")
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
import re


@Language.component("ioc_detector")
def ioc_detector(doc: Doc):
    # IOC detection logic
    iocs = {"IP": [], "DOMAIN": [], "HASH": []}
    for token in doc:
        if re.match(r"\b\d{1,3}(\.\d{1,3}){3}\b", token.text):  # Example for IP
            iocs["IP"].append(token.text)
    for label, values in iocs.items():
        spans = [
            Span(doc, doc.text.find(val), doc.text.find(val) + len(val), label=label)
            for val in values
        ]
        doc.ents = list(doc.ents) + spans
    return doc


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("ioc_detector", last=True)
import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning

# Suppress SSL warnings
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def summarize_text(text, model_name="t5-small", max_length=512):
    from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(
        text, max_length=max_length, truncation=True, return_tensors="tf"
    )
    summary_ids = model.generate(
        inputs["input_ids"], max_length=150, min_length=30, do_sample=False
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text_into_chunks(text, chunk_size=512):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])


def summarize_long_text(text, model_name="t5-small"):
    summaries = []
    for chunk in split_text_into_chunks(text, chunk_size=512):
        summary = summarize_text(chunk, model_name=model_name)
        summaries.append(summary)
    return " ".join(summaries)


def process_files_with_summary_incremental(dir_path, model_name="t5-small", output_file="summaries.txt"):
    """Summarize files incrementally and write to disk to reduce memory usage."""
    with open(output_file, "w", encoding="utf-8") as output:
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                text = extract_text_from_txt(file_path)
                summary = summarize_long_text(text, model_name=model_name)
                output.write(f"Summary for {filename}:\n{summary}\n\n")
                print(f"Processed {filename}")


if __name__ == "__main__":
    dir_path = "data/raw_reports"
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
    else:
        process_files_with_summary_incremental(dir_path)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_model(vocab_size, embedding_dim, input_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data, labels):
    model = build_model(vocab_size=5000, embedding_dim=100, input_length=200)
    model.fit(data, labels, epochs=10, validation_split=0.2)
    model.save("models/ioc_detection")
