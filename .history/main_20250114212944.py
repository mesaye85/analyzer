import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import spacy
from functools import lru_cache
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

# Use spaCy instead of NLTK for more efficient sentence tokenization
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])
except OSError:
    logger.info("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])

class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass

class ModelOperationError(Exception):
    """Custom exception for model operation errors."""
    pass

# ===============================
# Utility Functions
# ===============================

def validate_file(file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
    """
    Validate file before processing.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        bool: True if file is valid
        
    Raises:
        FileValidationError: If file validation fails
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileValidationError(f"File not found: {file_path}")
    
    if not path.is_file():
        raise FileValidationError(f"Not a file: {file_path}")
        
    if path.suffix.lower() != '.txt':
        raise FileValidationError(f"Invalid file type: {path.suffix}")
        
    if path.stat().st_size > max_size_mb * 1024 * 1024:
        raise FileValidationError(f"File too large: {file_path}")
        
    return True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    """
    Extract text content from a file with retry logic.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Extracted text content
        
    Raises:
        FileValidationError: If file operations fail
    """
    try:
        validate_file(file_path)
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
                
        raise FileValidationError(f"Unable to decode file with any supported encoding: {file_path}")
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise FileValidationError(f"Error reading file: {str(e)}")

# ===============================
# IOC Extraction
# ===============================

class IOCValidator:
    """Validator for different types of IOCs."""
    
    @staticmethod
    def validate_ip(ip: str) -> bool:
        """Validate IPv4 address."""
        try:
            parts = ip.split('.')
            return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
        except (AttributeError, TypeError, ValueError):
            return False
            
    @staticmethod
    def validate_domain(domain: str) -> bool:
        """Validate domain name."""
        if len(domain) > 255:
            return False
        if domain[-1] == ".":
            domain = domain[:-1]
        allowed = re.compile(r"^[a-zA-Z0-9-_]+(\.[a-zA-Z0-9-_]+)*$")
        return all(len(x) <= 63 for x in domain.split(".")) and bool(allowed.match(domain))

def extract_iocs(text: str) -> Dict[str, List[str]]:
    """
    Extract and validate IOCs using regex patterns.
    
    Args:
        text: Input text to extract IOCs from
        
    Returns:
        Dict containing validated IOCs by type
    """
    patterns = {
        "ips": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ipv6": r"\b([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b",
        "urls": r"https?://[-\w.%[\da-fA-F]{2}]+",
        "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "hashes": r"\b[a-fA-F0-9]{32,64}\b",
        "domains": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
    }

    iocs = {key: [] for key in patterns}
    
    for ioc_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        
        # Validate and deduplicate IOCs
        if ioc_type == "ips":
            iocs[ioc_type] = list(set(ip for ip in matches if IOCValidator.validate_ip(ip)))
        elif ioc_type == "domains":
            iocs[ioc_type] = list(set(domain for domain in matches if IOCValidator.validate_domain(domain)))
        else:
            iocs[ioc_type] = list(set(matches))
            
    return iocs

# ===============================
# Model Management
# ===============================

@lru_cache(maxsize=2)
def get_model(model_name: str):
    """Cache and retrieve models."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        raise ModelOperationError(f"Error loading model {model_name}: {str(e)}")

class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_classification(y_true, y_pred):
        """Calculate precision, recall, and F1 score."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @staticmethod
    def evaluate_summarization(original_texts, summaries):
        """Evaluate summarization quality."""
        # Add ROUGE score calculation here if needed
        pass

class MetricsDisplay:
    """Handle the display and logging of model metrics."""
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable string."""
        lines = [
            "=" * 50,
            "Model Evaluation Metrics",
            "=" * 50,
        ]
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            lines.append(f"{metric.replace('_', ' ').title()}: {formatted_value}")
            
        lines.append("=" * 50)
        return "\n".join(lines)
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], model_dir: str):
        """Save metrics to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(model_dir, f"metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_file}")
# ===============================
# Neural Network Model
# ===============================

def build_model(vocab_size: int = 5000,
                embedding_dim: int = 100,
                input_length: int = 200,
                lstm_units: int = 128) -> Sequential:
    """
    Build and compile the neural network model with configurable parameters.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimensionality of embeddings
        input_length: Length of input sequences
        lstm_units: Number of LSTM units
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size,
                 output_dim=embedding_dim,
                 input_length=input_length),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["accuracy", "precision", "recall"])
    
    return model

def train_model(data, labels, model_dir: str = "models"):
    """
    Train the neural network model with early stopping and checkpoints.
    
    Args:
        data: Training data
        labels: Training labels
        model_dir: Directory to save the model
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model = build_model()
    metrics_display = MetricsDisplay()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = model.fit(
        data,
        labels,
        epochs=10,
        validation_split=0.2,
        batch_size=32,
        callbacks=callbacks,
        verbose=1  # Show progress bar
    )
    
    # Evaluate model
    evaluator = ModelEvaluator()
    val_data = data[int(len(data) * 0.8):]
    val_labels = labels[int(len(labels) * 0.8):]
    predictions = model.predict(val_data) > 0.5
    
    # Collect all metrics
    metrics = {
        'classification_metrics': evaluator.evaluate_classification(val_labels, predictions),
        'training_history': {
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_precision': float(history.history['precision'][-1]),
            'final_recall': float(history.history['recall'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'epochs_trained': len(history.history['loss'])
        }
    }
    
    # Display metrics
    print(metrics_display.format_metrics(metrics))
    logger.info("Model Evaluation Results:\n" + metrics_display.format_metrics(metrics))
    
    # Save metrics
    metrics_display.save_metrics(metrics, model_dir)
    
    return model, metrics

# ===============================
# Main Execution
# ===============================

def main():
    """Main execution function."""
    try:
        raw_reports_dir = "raw_reports"
        Path("output").mkdir(parents=True, exist_ok=True)
        
        # Process files and extract IOCs
        results = []
        for filename in os.listdir(raw_reports_dir):
            try:
                file_path = os.path.join(raw_reports_dir, filename)
                text = extract_text_from_txt(file_path)
                iocs = extract_iocs(text)
                results.append({"file": filename, "iocs": iocs})
            except (FileValidationError, Exception) as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        # Save results
        if results:
            df = pd.DataFrame([
                {"file": item["file"], "ioc_type": ioc_type, "ioc": ioc}
                for item in results
                for ioc_type, iocs in item["iocs"].items()
                for ioc in iocs
            ])
            df.to_csv("output/extracted_iocs.csv", index=False)
            logger.info("Analysis completed successfully")
        else:
            logger.warning("No results were generated")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()