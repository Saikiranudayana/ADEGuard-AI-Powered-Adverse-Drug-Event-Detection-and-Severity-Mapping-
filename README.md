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
# EDA-Exploratory Data Analysis
<img width="482" height="502" alt="image" src="https://github.com/user-attachments/assets/d8d9abf7-31da-4c8d-ab22-8e391be875da" />

1. Gender Distribution (Pie Chart)

- Majority of reports are from Females (62.7%).

- Males contribute 32.5%, while Unknown gender accounts for 4.8%.

- This indicates a strong skew toward female-reported cases.

<img width="1032" height="371" alt="image" src="https://github.com/user-attachments/assets/bf45b8fe-54ad-4a01-8146-d381ca7a3be7" />


2. Age Distribution (Histogram)

- Reports are lowest in very young (0–9) and very old (100+).

- Peak reporting is observed between 50–69 years, especially in the 50–59 and 60–69 groups.

- Secondary peaks occur in 30–49 years, while reports decline after 70.

- Suggests middle-aged and elderly populations contribute the most reports.

<img width="720" height="545" alt="image" src="https://github.com/user-attachments/assets/61c075b5-5a71-4dca-b8d6-e4bbbb742a30" />

3. Heatmap of Reports by Sex and Age Group

- Confirms patterns from above but adds gender detail.

- Females dominate across all age groups, especially between 30–69 years, with peak counts in 60–69 (241k) and 50–59 (233k).

- Males show maximum counts in the same range (60–69: 120k; 50–59: 98k) but consistently fewer than females.

- nUnknown category is minimal but more frequent in younger and middle-aged groups.

- Overall, the 60–69 female group is the highest reporting cluster.
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/c52d3473-bb6f-4d78-b80a-d0dc51cb1d0d" />

4.Bar plot for number of cases per state

- California's significant lead in reporting numbers
- The concentration of reports in the top 5 states
- The geographic distribution showing coastal dominance
- The effectiveness of the visualization's color scheme
# 🎨 User Interface (UI) Component: The ADE Analysis Dashboard
The User Interface is the primary way users interact with the ADE analysis system. Built as a sleek, dark-themed dashboard, it provides an intuitive platform for leveraging Natural Language Processing (NLP) and Machine Learning techniques to understand Adverse Drug Events (ADE) from raw text.

🚀 Key Features and Components
The UI is structured around four main features, accessible via the navigation sidebar and the tabs within the analysis pages.

1. Home Dashboard (Welcome Screen)
<img width="1859" height="953" alt="Screenshot 2025-09-26 083157" src="https://github.com/user-attachments/assets/1438eb90-896c-4ece-a25d-e46a995ac46c" />


Purpose: The entry point to the application. It clearly introduces the project's goal (analyzing ADEs) and lists the core ML-driven functionalities.

Getting Started: Provides direct links to the main workflow pages: Text Analysis, Batch Analysis, and Model Explanation.

Models Used: Briefly mentions the underlying technologies, such as BioBERT, establishing credibility and context.

2. Text Analysis: Deep Dive into a Single Report
<img width="1872" height="946" alt="Screenshot 2025-09-26 083613" src="https://github.com/user-attachments/assets/2131fecd-2b20-4630-bb7c-e001ea6fd550" />

<img width="1351" height="693" alt="Screenshot 2025-09-26 083623" src="https://github.com/user-attachments/assets/37322b7f-067b-41d5-9ca6-68bbb20d4f4d" />

<img width="1384" height="614" alt="Screenshot 2025-09-26 083631" src="https://github.com/user-attachments/assets/1b7d7485-0fe2-4a80-8817-90bc71e372e4" />

<img width="1405" height="882" alt="Screenshot 2025-09-26 083644" src="https://github.com/user-attachments/assets/e04ca2ec-e507-4546-a8b5-2d0b2185f7af" />



This page is an interactive sandbox where a user can paste an individual text report (e.g., a patient note) and instantly see the results of the analysis.

Tab	Functionality	Visual Output
Entity Extraction	Identifies and highlights key medical entities in the input text.	A table showing extracted DRUG (e.g., ibuprofen), SYMPTOM (e.g., headache, nausea), and other relevant terms.
Severity Analysis	Classifies the overall severity of the adverse event.	A bar chart showing the probability distribution across severity levels (e.g., mild, moderate, severe) and the final Predicted Severity with a confidence score.
Symptom Clustering	Groups similar symptoms to identify patterns, even in a single report.	A list of Extracted Symptoms followed by Symptom Clusters (e.g., Cluster 0: headache, Cluster 1: nausea).
Explanation (XAI)	Uses Explainable AI (XAI) to show why the model made a specific severity prediction.	A bar chart showing Feature Importance (e.g., Patient, experienced, headache) where green bars indicate features that increase the probability of the predicted severity.

Export to Sheets
3. Batch Analysis: Processing Multiple Reports
<img width="1446" height="519" alt="Screenshot 2025-09-26 084051" src="https://github.com/user-attachments/assets/abe5e563-9a28-42a1-9942-7dbbc8763463" />
<img width="1444" height="689" alt="Screenshot 2025-09-26 083858" src="https://github.com/user-attachments/assets/151a6c11-8365-45f4-ab4e-bcd07b98b92d" />
<img width="1379" height="710" alt="Screenshot 2025-09-26 084056" src="https://github.com/user-attachments/assets/2a97f48c-9203-4555-974a-17b981254eda" />





This component is designed for high-throughput processing of large datasets (e.g., a CSV file).

Input & Settings: Allows the user to select the column containing ADE descriptions and specify the Number of records to analyze. It includes an execution button to trigger the Run Batch Analysis.

Analysis Results: Displays a table of processed records, showing the original text along with the predicted severity and confidence for each entry.

Summary Statistics: Provides a high-level overview of the batch results, such as a Severity Distribution pie chart showing the percentage of mild vs. severe events.

Data Export: Features a prominent "Download Results as CSV" button, allowing users to save the analyzed data for further external analysis.

4. Model Explanation: Performance Transparency
<img width="658" height="542" alt="Screenshot 2025-09-26 084216" src="https://github.com/user-attachments/assets/73a6c371-1210-4512-8b04-ce5a04e48d40" />

<img width="773" height="445" alt="Screenshot 2025-09-26 084220" src="https://github.com/user-attachments/assets/70731d07-5be4-46e8-a6c8-cae200588aa6" />



This page provides transparency into the underlying machine learning models used in the dashboard.

Entity Extraction Performance: Displays key metrics (Precision, Recall, F1-Score) broken down by different entity types (e.g., DRUG, SYMPTOM, DOSAGE), giving developers and users insight into the model's strengths and weaknesses.

Severity Classification Performance: Shows overall classification metrics (Accuracy, Precision, Recall, F1-Score) for the severity prediction model.

