import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import spacy
from functools import lru_cache, wraps
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("analyzer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Use spaCy instead of NLTK for more efficient sentence tokenization
@contextmanager
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        yield nlp
    except OSError:
        logger.info("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        yield nlp

# Custom Exceptions
class AnalyzerError(Exception):
    """Base exception for the analyzer."""
    pass

class FileValidationError(AnalyzerError):
    """File validation errors."""
    pass

class ModelOperationError(AnalyzerError):
    """Model operation errors."""
    pass

# Decorators
def handle_file_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise FileValidationError(str(e))
    return wrapper

@contextmanager
def log_operation(operation_name: str):
    logger.info(f"Starting {operation_name}")
    try:
        yield
        logger.info(f"{operation_name} completed successfully")
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        raise

# ===============================
# Utility Functions
# ===============================

@handle_file_errors
def validate_file(file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
    """Validate file before processing."""
    path = Path(file_path)
    if not path.is_file():
        raise FileValidationError(f"Invalid or missing file: {path}")
    if path.suffix.lower() != ".txt":
        raise FileValidationError(f"Invalid file type: {path.suffix}")
    if path.stat().st_size > max_size_mb * 1024 * 1024:
        raise FileValidationError(f"File too large: {path}")
    return True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@handle_file_errors
def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    """Extract text content from a file with retry logic."""
    validate_file(file_path)
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise FileValidationError(f"Unable to decode file with supported encodings: {file_path}")

# ===============================
# IOC Extraction
# ===============================

class IOCValidator:
    """Validator for different types of IOCs."""
    
    @staticmethod
    def validate_with_pattern(value: str, pattern: str, additional_check: Optional[Callable] = None) -> bool:
        """Generic validation method."""
        try:
            if not re.match(pattern, value):
                return False
            return additional_check(value) if additional_check else True
        except Exception:
            return False

    @staticmethod
    def check_ip_parts(ip: str) -> bool:
        """Additional check for IP addresses."""
        try:
            return all(0 <= int(part) <= 255 for part in ip.split("."))
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def check_domain_length(domain: str) -> bool:
        """Additional check for domain names."""
        return all(len(x) <= 63 for x in domain.split("."))

class IOCExtractor:
    """Handle IOC extraction and validation."""
    
    PATTERNS = {
        "ips": (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", IOCValidator.check_ip_parts),
        "ipv6": (r"\b([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b", None),
        "urls": (r"https?://[-\w.%[\da-fA-F]{2}]+", None),
        "emails": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", None),
        "hashes": (r"\b[a-fA-F0-9]{32,64}\b", None),
        "domains": (r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", IOCValidator.check_domain_length),
    }

    @classmethod
    def extract_iocs(cls, text: str) -> Dict[str, List[str]]:
        """Extract and validate IOCs using regex patterns."""
        iocs = {key: [] for key in cls.PATTERNS}
        for ioc_type, (pattern, additional_check) in cls.PATTERNS.items():
            matches = re.findall(pattern, text)
            iocs[ioc_type] = list(set(
                match for match in matches 
                if IOCValidator.validate_with_pattern(match, pattern, additional_check)
            ))
        return iocs

# ===============================
# Model Management
# ===============================

class ModelManager:
    """Handle model operations and caching."""
    
    @staticmethod
    @lru_cache(maxsize=2)
    def get_model(model_name: str):
        """Cache and retrieve models."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            raise ModelOperationError(f"Error loading model {model_name}: {str(e)}")

class MetricsManager:
    """Handle metrics collection and display."""
    
    @staticmethod
    def collect_training_metrics(history, evaluator, val_data, val_labels, model):
        """Collect all training and validation metrics."""
        predictions = model.predict(val_data) > 0.5
        return {
            "classification_metrics": evaluator.evaluate_classification(val_labels, predictions),
            "training_history": {
                metric: float(history.history[metric][-1])
                for metric in ["loss", "accuracy", "precision", "recall"]
            },
            "best_validation": {
                f"best_val_{metric}": float(
                    min(history.history[f"val_{metric}"])
                    if metric == "loss"
                    else max(history.history[f"val_{metric}"])
                )
                for metric in ["loss", "accuracy"]
            },
            "epochs_trained": len(history.history["loss"])
        }

    def format_and_save_metrics(self, metrics: Dict[str, Any], model_dir: str, prefix: str = ""):
        """Format metrics, display them, and save to file."""
        formatted_metrics = self._format_metrics(metrics)
        print(f"\n{prefix}Metrics:")
        print(formatted_metrics)
        logger.info(f"{prefix}Metrics:\n{formatted_metrics}")
        self._save_metrics(metrics, model_dir)

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable string."""
        lines = ["=" * 50, "Metrics Report", "=" * 50]
        for category, values in metrics.items():
            lines.append(f"\n{category.replace('_', ' ').title()}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    lines.append(f"  {key.replace('_', ' ').title()}: {formatted_value}")
            else:
                formatted_value = f"{values:.4f}" if isinstance(values, float) else str(values)
                lines.append(f"  {formatted_value}")
        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def _save_metrics(metrics: Dict[str, Any], model_dir: str):
        """Save metrics to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = Path(model_dir) / f"metrics_{timestamp}.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_file}")

# ===============================
# Neural Network Model
# ===============================

class ModelBuilder:
    """Handle neural network model creation and training."""
    
    @staticmethod
    def build_model(
        vocab_size: int = 5000,
        embedding_dim: int = 100,
        input_length: int = 200,
        lstm_units: int = 128,
    ) -> Sequential:
        """Build and compile the neural network model."""
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
            LSTM(lstm_units, return_sequences=True),
            LSTM(lstm_units),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )
        return model

    @staticmethod
    def train_model(data, labels, model_dir: str = "models"):
        """Train the neural network model with metrics collection."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        with log_operation("model training"):
            model = ModelBuilder.build_model()
            metrics_manager = MetricsManager()
            
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=str(Path(model_dir) / "best_model.h5"),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]
            
            history = model.fit(
                data,
                labels,
                epochs=10,
                validation_split=0.2,
                batch_size=32,
                callbacks=callbacks,
                verbose=1,
            )
            
            val_data = data[int(len(data) * 0.8):]
            val_labels = labels[int(len(labels) * 0.8):]
            
            metrics = metrics_manager.collect_training_metrics(
                history, ModelEvaluator(), val_data, val_labels, model
            )
            metrics_manager.format_and_save_metrics(metrics, model_dir)
            
            return model, metrics

# ===============================
# Main Execution
# ===============================

def process_files(raw_reports_dir: str) -> List[Dict]:
    """Process all files in the directory."""
    results = []
    with log_operation("file processing"):
        for filename in os.listdir(raw_reports_dir):
            try:
                file_path = Path(raw_reports_dir) / filename
                text = extract_text_from_txt(file_path)
                iocs = IOCExtractor.extract_iocs(text)
                results.append({"file": filename, "iocs": iocs})
            except FileValidationError as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    return results

def save_results(results: List[Dict], output_dir: str = "output"):
    """Save and display results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not results:
        logger.warning("No results were generated")
        return
    
    df = pd.DataFrame([
        {"file": item["file"], "ioc_type": ioc_type, "ioc": ioc}
        for item in results
        for ioc_type, iocs in item["iocs"].items()
        for ioc in iocs
    ])
    
    df.to_csv(Path(output_dir) / "extracted_iocs.csv", index=False)
    
    metrics = {
        "total_files_processed": len(results),
        "total_iocs_found": len(df),
        "iocs_by_type": df["ioc_type"].value_counts().to_dict(),
        "files_with_iocs": df["file"].nunique(),
    }
    
    MetricsManager().format_and_save_metrics(metrics, output_dir, "IOC Extraction ")

def main():
    """Main execution function."""
    try:
        with log_operation("main execution"):
            raw_reports_dir = "raw_reports"
            results = process_files(raw_reports_dir)
            save_results(results)
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()