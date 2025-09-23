"""
Script for preparing training data for ADE models.

This script demonstrates how to:
1. Process raw data (like VAERS or other medical data)
2. Clean and preprocess text
3. Create annotation tasks for Label Studio
4. Convert annotated data for model training

Usage:
    python scripts/prepare_training_data.py --input data/raw_data.csv --output data/processed/
"""

import os
import sys
import pandas as pd
import json
import re
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add the project root to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text


def extract_ade_candidates(df: pd.DataFrame, text_col: str, min_length: int = 10, max_length: int = 500) -> pd.DataFrame:
    """
    Extract potential ADE descriptions from raw data.
    
    Args:
        df: Input dataframe
        text_col: Name of column containing text
        min_length: Minimum character length of descriptions
        max_length: Maximum character length of descriptions
        
    Returns:
        Dataframe with candidate ADE descriptions
    """
    logger.info("Extracting ADE candidates")
    
    # Check text column exists
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataframe")
    
    # Clean text
    df['cleaned_text'] = df[text_col].fillna('').apply(clean_text)
    
    # Filter by length
    df = df[(df['cleaned_text'].str.len() >= min_length) & 
            (df['cleaned_text'].str.len() <= max_length)]
    
    logger.info(f"Extracted {len(df)} candidate descriptions")
    
    return df


def preprocess_for_annotation(df: pd.DataFrame, text_col: str, 
                              additional_cols: Optional[List[str]] = None) -> List[Dict]:
    """
    Prepare data for annotation in Label Studio.
    
    Args:
        df: Input dataframe with candidate descriptions
        text_col: Name of column containing text
        additional_cols: Additional columns to include in metadata
        
    Returns:
        List of annotation tasks in Label Studio format
    """
    logger.info("Preparing data for annotation")
    
    if additional_cols is None:
        additional_cols = []
    
    tasks = []
    for _, row in df.iterrows():
        task = {
            "data": {
                "text": row[text_col]
            },
            "meta": {}
        }
        
        # Add additional columns as metadata
        for col in additional_cols:
            if col in row:
                task["meta"][col] = str(row[col])
        
        tasks.append(task)
    
    logger.info(f"Created {len(tasks)} annotation tasks")
    
    return tasks


def create_label_studio_file(tasks: List[Dict], output_file: str):
    """
    Create Label Studio tasks JSON file.
    
    Args:
        tasks: List of Label Studio tasks
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    
    logger.info(f"Saved {len(tasks)} annotation tasks to {output_file}")


def create_label_studio_config(entity_types: List[str], output_file: str):
    """
    Create Label Studio configuration file with entity types.
    
    Args:
        entity_types: List of entity types
        output_file: Path to output file
    """
    # Create labels configuration
    labels = [{"value": entity_type, "background": get_color_for_entity(entity_type)} 
              for entity_type in entity_types]
    
    config = {
        "interfaces": [
            "panel",
            "update",
            "controls",
            "side-column",
            "annotations:menu",
            "annotations:add-new",
            "annotations:delete",
            "predictions:menu"
        ],
        "editor": {
            "label": "Named Entity Recognition",
            "type": "ner"
        },
        "labels": labels,
        "relationMode": "relation",
        "showOverlap": False
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved Label Studio configuration to {output_file}")


def get_color_for_entity(entity_type: str) -> str:
    """
    Get color for entity type.
    
    Args:
        entity_type: Entity type
        
    Returns:
        Hex color code
    """
    # Define colors for common entity types
    entity_colors = {
        "DRUG": "#ff9900",
        "SYMPTOM": "#33cc33",
        "SEVERITY": "#ff3333",
        "DOSAGE": "#9966ff",
        "DURATION": "#3399ff",
        "FREQUENCY": "#ffcc00",
        "ROUTE": "#cc66ff"
    }
    
    return entity_colors.get(entity_type, "#999999")


def prepare_severity_data(df: pd.DataFrame, text_col: str, severity_keywords: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Automatically label texts with severity based on keywords.
    
    Args:
        df: Input dataframe
        text_col: Name of column with text
        severity_keywords: Dictionary mapping severity levels to keywords
        
    Returns:
        Dataframe with severity labels
    """
    logger.info("Preparing severity data")
    
    # Function to determine severity based on keywords
    def get_severity(text):
        text_lower = text.lower()
        
        # Check for each severity level
        for severity, keywords in severity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return severity
        
        # Default to unknown if no keywords match
        return "unknown"
    
    # Apply severity labeling
    df['severity'] = df[text_col].apply(get_severity)
    
    # Filter out unknown severity
    df_labeled = df[df['severity'] != "unknown"].copy()
    
    logger.info(f"Labeled {len(df_labeled)} examples with severity")
    
    return df_labeled


def extract_symptoms_from_text(df: pd.DataFrame, text_col: str, 
                               symptom_dictionary: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract symptoms from text based on a dictionary of symptoms.
    
    Args:
        df: Input dataframe
        text_col: Name of column with text
        symptom_dictionary: List of known symptom terms
        
    Returns:
        Dataframe with extracted symptoms and list of all found symptoms
    """
    logger.info("Extracting symptoms from text")
    
    # Compile regex pattern for symptoms
    # Use word boundaries to match whole words
    pattern = re.compile(r'\b(' + '|'.join(re.escape(s) for s in symptom_dictionary) + r')\b', re.IGNORECASE)
    
    # Function to extract symptoms from text
    def extract_symptoms(text):
        if not isinstance(text, str):
            return []
        
        matches = pattern.findall(text.lower())
        return list(set(matches))  # Remove duplicates
    
    # Extract symptoms from text
    df['symptoms'] = df[text_col].apply(extract_symptoms)
    
    # Count symptoms across all texts
    symptom_counts = defaultdict(int)
    for symptoms in df['symptoms']:
        for symptom in symptoms:
            symptom_counts[symptom] += 1
    
    # Sort symptoms by count
    all_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"Extracted {len(all_symptoms)} unique symptoms")
    
    return df, [symptom for symptom, _ in all_symptoms]


def load_symptom_dictionary(dictionary_file: Optional[str] = None) -> List[str]:
    """
    Load symptom dictionary from file or use default medical terms.
    
    Args:
        dictionary_file: Path to dictionary file (one term per line)
        
    Returns:
        List of symptom terms
    """
    if dictionary_file and os.path.exists(dictionary_file):
        with open(dictionary_file, 'r', encoding='utf-8') as f:
            symptom_dict = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symptom_dict)} symptoms from dictionary file")
        return symptom_dict
    
    # Default list of common medical symptoms/conditions
    default_symptoms = [
        "headache", "nausea", "vomiting", "fever", "pain", "dizziness",
        "fatigue", "rash", "swelling", "itching", "cough", "shortness of breath",
        "diarrhea", "constipation", "blurred vision", "chest pain", "abdominal pain",
        "insomnia", "anxiety", "depression", "confusion", "weakness", "numbness",
        "tingling", "seizure", "tremor", "muscle pain", "joint pain", "back pain",
        "throat pain", "ear pain", "eye pain", "chills", "sweating", "loss of appetite",
        "weight loss", "weight gain", "edema", "hypertension", "hypotension",
        "tachycardia", "bradycardia", "arrhythmia", "palpitations", "syncope",
        "dyspnea", "wheezing", "hemoptysis", "jaundice", "hematuria", "dysuria",
        "polyuria", "oliguria", "anuria", "hematochezia", "melena", "hematemesis",
        "dysphagia", "odynophagia", "heartburn", "indigestion", "bloating", "flatulence",
        "pruritus", "eczema", "urticaria", "petechiae", "ecchymosis", "paresthesia",
        "paralysis", "ataxia", "aphasia", "dysarthria", "diplopia", "photophobia",
        "tinnitus", "vertigo", "epistaxis", "amenorrhea", "menorrhagia", "dysmenorrhea",
        "anaphylaxis", "dyspepsia"
    ]
    
    logger.info(f"Using default symptom dictionary with {len(default_symptoms)} terms")
    
    return default_symptoms


def main():
    """Main function to demonstrate data preparation workflow."""
    parser = argparse.ArgumentParser(description="Prepare ADE training data")
    parser.add_argument("--input", type=str, required=True, help="Input data file (CSV)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--text-col", type=str, default="text", help="Name of column containing text")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum text length")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum text length")
    parser.add_argument("--symptom-dict", type=str, help="Path to symptom dictionary file")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load input data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    # Extract ADE candidates
    candidates_df = extract_ade_candidates(
        df, 
        text_col=args.text_col,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Prepare NER annotation tasks
    annotation_tasks = preprocess_for_annotation(
        candidates_df,
        text_col='cleaned_text',
        additional_cols=['id']  # Include original ID if available
    )
    
    # Save Label Studio tasks
    create_label_studio_file(
        tasks=annotation_tasks,
        output_file=os.path.join(args.output, "label_studio_tasks.json")
    )
    
    # Create Label Studio configuration
    entity_types = ["DRUG", "SYMPTOM", "SEVERITY", "DOSAGE", "DURATION", "FREQUENCY", "ROUTE"]
    create_label_studio_config(
        entity_types=entity_types,
        output_file=os.path.join(args.output, "label_studio_config.json")
    )
    
    # Prepare severity data
    severity_keywords = {
        "mild": ["mild", "slight", "minor", "minimal", "light", "low", "small"],
        "moderate": ["moderate", "medium", "intermediate", "average"],
        "severe": ["severe", "serious", "extreme", "high", "intense", "major", "critical", "life-threatening", "lethal"]
    }
    
    severity_df = prepare_severity_data(
        df=candidates_df,
        text_col='cleaned_text',
        severity_keywords=severity_keywords
    )
    
    # Save severity data
    severity_df[['cleaned_text', 'severity']].to_csv(
        os.path.join(args.output, "severity_data.csv"),
        index=False
    )
    
    # Load symptom dictionary
    symptom_dict = load_symptom_dictionary(args.symptom_dict)
    
    # Extract symptoms from text
    symptom_df, all_symptoms = extract_symptoms_from_text(
        df=candidates_df,
        text_col='cleaned_text',
        symptom_dictionary=symptom_dict
    )
    
    # Save symptoms data
    pd.DataFrame({'symptom': all_symptoms}).to_csv(
        os.path.join(args.output, "symptoms.csv"),
        index=False
    )
    
    # Save summary of extracted data
    summary = {
        "total_records": len(df),
        "candidate_descriptions": len(candidates_df),
        "labeled_severity_examples": len(severity_df),
        "unique_symptoms": len(all_symptoms),
        "entity_types": entity_types
    }
    
    with open(os.path.join(args.output, "data_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Data preparation completed")
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
