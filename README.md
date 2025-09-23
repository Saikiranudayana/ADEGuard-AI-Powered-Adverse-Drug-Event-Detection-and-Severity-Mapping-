# Adverse Drug Event Detection System

This project implements an advanced Natural Language Processing system for detecting, classifying, and analyzing Adverse Drug Events (ADEs) from medical text. It uses state-of-the-art deep learning models and machine learning techniques to extract entities, classify severity, cluster symptoms, and provide model explanations.

## Features

- **ADE Entity Extraction**: Identifies drugs, symptoms, severity indicators, and other medical entities in text using BioBERT
- **Severity Classification**: Automatically categorizes adverse events as mild, moderate, or severe
- **Symptom Clustering**: Groups similar symptoms to identify patterns using HDBSCAN and Sentence-BERT
- **Explainable AI**: Provides insight into model decisions using LIME/SHAP

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── data/                   # Data directory for datasets
├── models/                 # Directory for trained models
│   ├── ner/                # Entity extraction model
│   └── severity/           # Severity classification model
├── scripts/                # Utility scripts
│   ├── inference.py        # Script for real-time ADE detection
│   ├── prepare_training_data.py  # Script for data preparation
│   └── train_and_evaluate.py     # Script for training models
├── src/                    # Source code
│   └── training/           # Model training code
│       └── model.py        # Core model implementation
└── requirements.txt        # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

To prepare training data from raw medical texts:

```bash
python scripts/prepare_training_data.py --input data/raw_data.csv --output data/processed/
```

This script will:
- Clean and preprocess the text
- Create annotation tasks for Label Studio
- Generate preliminary severity labels
- Extract symptoms for clustering

### Training Models

To train the ADE models:

```bash
python scripts/train_and_evaluate.py --ner-annotations data/processed/label_studio_annotations.json --severity-file data/processed/severity_data.csv --symptoms-file data/processed/symptoms.csv --model-dir models/
```

This script will:
- Train the entity extraction model
- Train the severity classification model
- Perform symptom clustering
- Generate and evaluate model explanations

### Running Inference

To analyze text for ADEs:

```bash
python scripts/inference.py --text "Patient experienced severe headache after taking aspirin." --model-dir models/ --output results.json
```

For batch processing:

```bash
python scripts/inference.py --file data/test_cases.txt --model-dir models/ --output batch_results.json
```

### Running the Web Application

To launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501.

## Model Architecture

### ADE Entity Extractor
- Based on BioBERT with a token classification head
- Fine-tuned for Named Entity Recognition on medical text
- Entity types: DRUG, SYMPTOM, SEVERITY, DOSAGE, DURATION, FREQUENCY, ROUTE

### ADE Severity Classifier
- Text classification model using biomedical embeddings
- Categorizes severity as mild, moderate, or severe
- Returns confidence scores and probability distribution

### ADE Symptom Clusterer
- Uses Sentence-BERT to generate embeddings for symptoms
- UMAP for dimensionality reduction
- HDBSCAN for density-based clustering
- Identifies groups of semantically similar symptoms

### ADE Explainer
- Uses LIME/SHAP for model interpretability
- Identifies which words/phrases most influence predictions
- Helps medical professionals understand model reasoning

## Contributing

Contributions to improve the ADE detection system are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
