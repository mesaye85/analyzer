"""Experimental LSTM report classifier. NOT part of the IOC tool.

This code was in main.py from the start but was never wired into the pipeline:
nothing calls it, and there is no labeled dataset to train or evaluate it on.
It is kept here for reference only. The CLI (ioc_analyzer/) does not import it.

To experiment with it you need the packages in requirements-ml.txt and a
labeled set of (tokenized report, 0/1 label) pairs, which this repo does not
include. The unused spaCy loader and tenacity retry helpers from the original
file were dropped, imports were consolidated on keras, and the checkpoint now
uses the .keras format Keras 3 expects; the model, metrics and evaluation code
is otherwise as it was.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelOperationError(Exception):
    """Model operation errors."""


@contextmanager
def log_operation(operation_name: str):
    logger.info("Starting %s", operation_name)
    try:
        yield
        logger.info("%s completed successfully", operation_name)
    except Exception as e:
        logger.error("%s failed: %s", operation_name, e)
        raise


class ModelManager:
    """Load and cache a Hugging Face seq2seq model (e.g. T5 for summarization)."""

    @staticmethod
    @lru_cache(maxsize=2)
    def get_model(model_name: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            raise ModelOperationError(f"Error loading model {model_name}: {e}")


class ModelEvaluator:
    @staticmethod
    def evaluate_classification(true_labels, predictions):
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
        return {"precision": precision, "recall": recall, "f1_score": f1}


class MetricsManager:
    """Collect, print and save training metrics."""

    @staticmethod
    def collect_training_metrics(history, evaluator, val_data, val_labels, model):
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
            "epochs_trained": len(history.history["loss"]),
        }

    def format_and_save_metrics(self, metrics: Dict[str, Any], model_dir: str, prefix: str = ""):
        formatted = self._format_metrics(metrics)
        print(f"\n{prefix}Metrics:")
        print(formatted)
        self._save_metrics(metrics, model_dir)

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        lines = ["=" * 50, "Metrics Report", "=" * 50]
        for category, values in metrics.items():
            lines.append(f"\n{category.replace('_', ' ').title()}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    shown = f"{value:.4f}" if isinstance(value, float) else str(value)
                    lines.append(f"  {key.replace('_', ' ').title()}: {shown}")
            else:
                shown = f"{values:.4f}" if isinstance(values, float) else str(values)
                lines.append(f"  {shown}")
        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def _save_metrics(metrics: Dict[str, Any], model_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = Path(model_dir) / f"metrics_{timestamp}.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved to %s", metrics_file)


class ModelBuilder:
    """Embedding + two LSTM layers + binary output."""

    @staticmethod
    def build_model(
        vocab_size: int = 5000,
        embedding_dim: int = 100,
        input_length: int = 200,
        lstm_units: int = 128,
    ) -> Sequential:
        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
                LSTM(lstm_units, return_sequences=True),
                LSTM(lstm_units),
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"])
        return model

    @staticmethod
    def train_model(data, labels, model_dir: str = "models"):
        """Train on (data, labels). Requires a labeled dataset this repo does not have."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        with log_operation("model training"):
            model = ModelBuilder.build_model()
            metrics_manager = MetricsManager()
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=str(Path(model_dir) / "best_model.keras"),
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

            split = int(len(data) * 0.8)
            val_data, val_labels = data[split:], labels[split:]

            metrics = metrics_manager.collect_training_metrics(history, ModelEvaluator(), val_data, val_labels, model)
            metrics_manager.format_and_save_metrics(metrics, model_dir)
            return model, metrics
