# Standard library imports
import argparse
import re
import logging
from typing import Dict, Set, Union
from pathlib import Path
from dataclasses import dataclass

# Third-party imports
import spacy
import pandas as pd
import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    # Version compatibility check
    tf_version = tf.__version__
    if tf_version < "2.0.0":
        raise ImportError("This project requires TensorFlow 2.0 or higher")
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning(
        "TensorFlow not found. Please install with: pip install tensorflow>=2.0.0"
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IOCConfig:
    """Configuration for IOC detection patterns and settings."""
    patterns: Dict[str, str] = None

    def __post_init__(self):
        self.patterns = {
            "ip": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
            "md5": r"\b[a-fA-F0-9]{32}\b",
            "sha1": r"\b[a-fA-F0-9]{40}\b",
            "sha256": r"\b[a-fA-F0-9]{64}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[\w./?%&=-]*)?"

        }


class PDFExtractor:
    """Handles PDF text extraction with error handling and validation."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def extract_text(self) -> str:
        """Extract text from PDF with proper error handling."""
        try:
            import PyPDF2
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return "\n".join(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise


class TextPreprocessor:
    """Handles text cleaning and normalization."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove unnecessary whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize quotes
        text = re.sub(r'[""'']', '"', text)
        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char == '\n')
        return text.strip()


class IOCExtractor:
    """Extracts Indicators of Compromise (IOCs) from text."""

    def __init__(self, config: IOCConfig = None):
        self.config = config or IOCConfig()
        self._load_spacy_model()

    def _load_spacy_model(self):
        """Load and configure spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.add_pipe("ioc_detector", config={"patterns": self.config.patterns})
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

    def extract_iocs(self, text: str) -> Dict[str, Set[str]]:
        """Extract IOCs using both regex and NLP approaches."""
        iocs = {ioc_type: set() for ioc_type in self.config.patterns.keys()}

        # Regex-based extraction
        for ioc_type, pattern in self.config.patterns.items():
            matches = re.findall(pattern, text)
            iocs[ioc_type].update(matches)

        # NLP-based extraction
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in iocs:
                iocs[ent.label_].add(ent.text)

        return iocs


@Language.factory("ioc_detector")
def create_ioc_detector(nlp, name, patterns):
    return IOCDetector(nlp, patterns)

class IOCDetector:
    def __init__(self, nlp, patterns):
        self.nlp = nlp
        self.patterns = patterns

    def __call__(self, doc):
        # Your IOC detection logic here
        return doc

# Load spaCy model and add the custom component
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("ioc_detector", config={"patterns": IOCConfig().patterns})

# Example usage
doc = nlp("Example text with an IP address 192.168.1.1")


class ThreatIntelModel:
    """Neural network model for IOC classification and threat intelligence."""

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 100, input_length: int = 200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build and compile the neural network model."""
        model = tf.keras.Sequential([
            Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_dim,
                      input_length=self.input_length),
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
            Bidirectional(LSTM(64, dropout=0.2)),
            Dense(64, activation='relu'),
            Dense(len(IOCConfig().patterns), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def train(self,
              data: np.ndarray,
              labels: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 10,
              batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the model with early stopping and checkpoints."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        return self.model.fit(
            data,
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )


def sum_even_numbers(numbers):
    """
    Calculate sum of even numbers in a list
    Args:
        numbers (list): List of integers
    Returns:
        int: Sum of even numbers
    """
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total

# Example usage
if __name__ == "__main__":
    test_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    result = sum_even_numbers(test_numbers)
    print(f"Sum of even numbers: {result}")  # Should output: 20 (2+4+6+8)


def main():
    """Main execution function."""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", action="store_true", help="Run in test mode with sample data")
        args = parser.parse_args()

        if args.test:
            logger.info("Running in test mode.")
            test_text = "Sample text with IP 192.168.1.1 and URL https://example.com."
            config = IOCConfig()
            ioc_extractor = IOCExtractor(config)
            iocs = ioc_extractor.extract_iocs(test_text)
            print(f"Extracted IOCs: {iocs}")
            return

        # Initialize components
        config = IOCConfig()
        pdf_extractor = PDFExtractor("data/raw_reports/sample.pdf")
        text_processor = TextPreprocessor()
        ioc_extractor = IOCExtractor(config)

        # Process document
        raw_text = pdf_extractor.extract_text()
        logger.info("Extracted raw text length: %d", len(raw_text))
        cleaned_text = text_processor.clean_text(raw_text)
        logger.info("Cleaned text length: %d", len(cleaned_text))
        iocs = ioc_extractor.extract_iocs(cleaned_text)

        # Save results
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame([
            {"type": ioc_type, "value": value}
            for ioc_type, values in iocs.items()
            for value in values
        ])
        results_df.to_csv(output_dir / "extracted_iocs.csv", index=False)
        logger.info("Successfully processed document. Found %d IOCs.", len(results_df))

    except Exception as e:
        logger.error("Error in main execution: %s", str(e))
        raise


if __name__ == "__main__":
    main()
