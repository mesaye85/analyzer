Analyzer

Analyzer is a robust Python toolkit for processing textual files, extracting Indicators of Compromise (IOCs), and training neural network models for classification tasks. It features a modular architecture with extensive logging, error handling, and metrics reporting for effective cybersecurity and text analytics workflows.

Features

IOC Extraction: Scans files and extracts IOCs such as IP addresses, domains, URLs, emails, and hashes using advanced regular expressions and validation heuristics.
Neural Network Model Training: Build and train LSTM-based models for binary classification tasks with Keras, including callbacks for early stopping and checkpointing.
Metrics Reporting: Collects and saves detailed training and extraction metrics in both human-readable and JSON formats.
Robust File Handling: Validates file types, sizes, and encodings; handles errors gracefully and logs all operations.
Caching & Retry Logic: Efficient model loading with caching and robust retry logic for file reading.
Extensive Logging: All operations are logged to both console and file.
Requirements

Python 3.7+
pip install -r requirements.txt
Main Dependencies

pandas
scikit-learn
keras, tensorflow
transformers
spacy
tenacity
Installation

Clone the repository:
bash
git clone https://github.com/mesaye85/analyzer.git
cd analyzer
Install dependencies:
bash
pip install -r requirements.txt
Download the spaCy English model (if not downloaded automatically):
bash
python -m spacy download en_core_web_sm
Usage

1. Prepare Input Data

Place your .txt files containing raw reports into the raw_reports/ directory.
Only .txt files under 100 MB are processed.
2. Run the Analyzer

bash
python main.py
3. Outputs

All extracted IOCs are saved in output/extracted_iocs.csv.
Metrics and summary statistics are saved in the output/ directory as JSON files and also logged.
4. Logs

Operation logs are saved in analyzer.log.
Project Structure

Code
analyzer/
│
├── main.py            # Main application file
├── requirements.txt   # Python dependencies
├── raw_reports/       # Directory for input .txt files
├── output/            # Results and metrics output
└── analyzer.log       # Log file
Components Overview

IOC Extraction:
Uses regular expressions and additional validation for accuracy. Supported types: IPs (v4/v6), domains, URLs, emails, hashes.

Model Training:

Builds a two-layer LSTM with embedding for sequence classification.
Trained on provided data with early stopping and checkpointing.
Reports training/validation metrics.
Metrics Management:

Aggregates and formats metrics.
Saves metrics as JSON with timestamped filenames.
Error Handling & Logging:

Custom exceptions for file and model operations.
All errors, warnings, and info events are logged.
Extending

Add new IOC patterns in the IOCExtractor.PATTERNS dictionary in main.py.
Modify ModelBuilder for different neural network architectures.
Integrate with other input formats by extending the file validation and extraction logic.
Troubleshooting

If spaCy model is missing, it will be downloaded automatically. If not, run:
bash
python -m spacy download en_core_web_sm
Ensure all input files are .txt and under the size limit.
Check analyzer.log for detailed error messages and stack traces.
License

MIT License

Author

Mesaye Addisu


