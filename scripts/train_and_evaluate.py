"""
Script for training and evaluating ADE models.

This script demonstrates how to:
1. Create and process annotated data
2. Train the entity extractor
3. Train the severity classifier
4. Cluster symptoms
5. Generate explanations

Usage:
    python scripts/train_and_evaluate.py
"""
import os
import sys
import pandas as pd
import json
import argparse
from typing import List, Dict
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

# Add the project root to the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ADE models
from src.training.model import (
    ADEEntityExtractor,
    ADESeverityClassifier,
    ADESymptomClusterer,
    ADEExplainer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_ner_dataset_from_label_studio(annotations_file: str):
    """
    Convert Label Studio annotations to HuggingFace dataset format.
    
    Args:
        annotations_file: Path to Label Studio JSON export
        
    Returns:
        Training and validation datasets
    """
    logger.info(f"Processing annotations from {annotations_file}")
    
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
        
    # Process annotations
    texts = []
    tags = []
    
    for annotation in annotations:
        if 'result' not in annotation or not annotation['result']:
            continue
            
        # Get text
        text = annotation.get('data', {}).get('text', '')
        if not text:
            continue
            
        # Create tags array (BIO scheme)
        text_tags = ['O'] * len(text)
        
        # Process each entity annotation
        for result in annotation['result']:
            if 'value' not in result:
                continue
                
            value = result['value']
            if 'labels' not in value or not value['labels']:
                continue
                
            # Extract entity info
            entity_type = value['labels'][0]  # Take first label
            start = value['start']
            end = value['end']
            
            # Apply BIO tagging
            if start < len(text_tags):
                text_tags[start] = f'B-{entity_type}'
                
            for i in range(start + 1, min(end, len(text_tags))):
                text_tags[i] = f'I-{entity_type}'
                
        texts.append(text)
        tags.append(text_tags)
    
    logger.info(f"Processed {len(texts)} annotated examples")
    
    # Create datasets dictionary
    from datasets import Dataset
    import numpy as np
    
    # Split into train/val
    split_idx = int(len(texts) * 0.8)
    
    train_dataset = Dataset.from_dict({
        'text': texts[:split_idx],
        'tags': tags[:split_idx]
    })
    
    val_dataset = Dataset.from_dict({
        'text': texts[split_idx:],
        'tags': tags[split_idx:]
    })
    
    logger.info(f"Created training dataset with {len(train_dataset)} examples")
    logger.info(f"Created validation dataset with {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


def prepare_severity_dataset_from_csv(csv_file: str, text_col: str, severity_col: str):
    """
    Prepare severity classification dataset from CSV file.
    
    Args:
        csv_file: Path to CSV file with ADE descriptions and severity labels
        text_col: Name of the column containing ADE text
        severity_col: Name of the column containing severity labels
        
    Returns:
        Training and validation datasets
    """
    logger.info(f"Loading severity data from {csv_file}")
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Check columns exist
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV")
    if severity_col not in df.columns:
        raise ValueError(f"Severity column '{severity_col}' not found in CSV")
        
    # Map severity labels to integers
    severity_map = {
        "mild": 0,
        "moderate": 1, 
        "severe": 2
    }
    
    # Convert labels to lowercase and map to integers
    df['label'] = df[severity_col].str.lower().map(severity_map)
    
    # Check if any labels couldn't be mapped
    unknown_labels = df[~df['label'].isin(severity_map.values())][severity_col].unique()
    if len(unknown_labels) > 0:
        logger.warning(f"Unknown severity labels: {unknown_labels}")
        
    # Drop rows with unknown labels
    df = df.dropna(subset=['label'])
    
    # Create datasets
    from datasets import Dataset
    
    # Split into train/val
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_dataset = Dataset.from_dict({
        'text': train_df[text_col].tolist(),
        'label': train_df['label'].astype(int).tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_df[text_col].tolist(),
        'label': val_df['label'].astype(int).tolist()
    })
    
    logger.info(f"Created training dataset with {len(train_dataset)} examples")
    logger.info(f"Created validation dataset with {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


def train_entity_extractor(model_dir: str, annotations_file: str = None):
    """
    Train the entity extractor model.
    
    Args:
        model_dir: Directory to save the trained model
        annotations_file: Path to Label Studio annotations file
    
    Returns:
        Trained ADEEntityExtractor
    """
    logger.info("Initializing entity extractor model")
    entity_extractor = ADEEntityExtractor()
    
    if annotations_file:
        logger.info("Training entity extractor with annotated data")
        train_dataset, val_dataset = prepare_ner_dataset_from_label_studio(annotations_file)
        
        # Create output directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Train model
        entity_extractor.train(
            train_data=train_dataset,
            val_data=val_dataset,
            output_dir=model_dir,
            epochs=3
        )
        
        logger.info(f"Entity extractor trained and saved to {model_dir}")
    else:
        logger.info("No annotations file provided, using pre-initialized model")
    
    return entity_extractor


def train_severity_classifier(model_dir: str, severity_file: str = None):
    """
    Train the severity classifier model.
    
    Args:
        model_dir: Directory to save the trained model
        severity_file: Path to CSV file with severity labels
    
    Returns:
        Trained ADESeverityClassifier
    """
    logger.info("Initializing severity classifier model")
    severity_classifier = ADESeverityClassifier()
    
    if severity_file:
        logger.info("Training severity classifier with labeled data")
        train_dataset, val_dataset = prepare_severity_dataset_from_csv(
            severity_file, 
            text_col="description", 
            severity_col="severity"
        )
        
        # Create output directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Train model
        severity_classifier.train(
            train_data=train_dataset,
            val_data=val_dataset,
            output_dir=model_dir,
            epochs=3
        )
        
        logger.info(f"Severity classifier trained and saved to {model_dir}")
    else:
        logger.info("No severity file provided, using pre-initialized model")
    
    return severity_classifier


def cluster_symptoms(symptoms_file: str = None):
    """
    Cluster symptoms based on semantic similarity.
    
    Args:
        symptoms_file: Path to CSV file with symptoms
    
    Returns:
        ADESymptomClusterer and clustering results
    """
    logger.info("Initializing symptom clusterer")
    symptom_clusterer = ADESymptomClusterer()
    
    if symptoms_file:
        logger.info(f"Loading symptoms from {symptoms_file}")
        
        # Load symptoms from CSV
        df = pd.read_csv(symptoms_file)
        
        # Check if age_group and modifier columns exist
        has_age_group = 'age_group' in df.columns
        has_modifier = 'modifier' in df.columns
        
        # Get symptoms and optional attributes
        symptoms = df['symptom'].tolist()
        age_groups = df['age_group'].tolist() if has_age_group else None
        modifiers = df['modifier'].tolist() if has_modifier else None
        
        # Cluster symptoms
        logger.info(f"Clustering {len(symptoms)} symptoms")
        clusters = symptom_clusterer.cluster_symptoms(
            symptoms=symptoms,
            age_groups=age_groups,
            modifiers=modifiers
        )
        
        logger.info(f"Found {clusters['num_clusters']} symptom clusters")
        
        # Visualize clusters if visualization data is available
        if 'visualization_data' in clusters:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Extract visualization data
                vis_data = clusters['visualization_data']
                x = [p['x'] for p in vis_data]
                y = [p['y'] for p in vis_data]
                labels = [p['cluster'] for p in vis_data]
                
                # Create plot
                plt.figure(figsize=(10, 8))
                
                # Get unique clusters
                unique_clusters = set(labels)
                
                # Create colormap
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
                
                # Plot each cluster
                for i, cluster_id in enumerate(unique_clusters):
                    if cluster_id == -1:
                        # Plot noise points in black
                        cluster_points = [(x[j], y[j]) for j in range(len(labels)) if labels[j] == -1]
                        if cluster_points:
                            cluster_x, cluster_y = zip(*cluster_points)
                            plt.scatter(cluster_x, cluster_y, c='black', label='Noise', alpha=0.5)
                    else:
                        # Plot cluster points
                        cluster_points = [(x[j], y[j]) for j in range(len(labels)) if labels[j] == cluster_id]
                        if cluster_points:
                            cluster_x, cluster_y = zip(*cluster_points)
                            plt.scatter(cluster_x, cluster_y, c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7)
                
                plt.title('Symptom Clusters')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                plt.legend()
                
                # Save plot
                plt.savefig('symptom_clusters.png')
                logger.info("Saved cluster visualization to symptom_clusters.png")
                
            except Exception as e:
                logger.error(f"Error visualizing clusters: {str(e)}")
        
        return symptom_clusterer, clusters
    else:
        logger.info("No symptoms file provided")
        return symptom_clusterer, None


def generate_explanations(text_examples: List[str], entity_extractor, severity_classifier):
    """
    Generate explanations for a list of text examples.
    
    Args:
        text_examples: List of text examples to explain
        entity_extractor: Trained entity extractor model
        severity_classifier: Trained severity classifier model
        
    Returns:
        List of explanations for each text example
    """
    logger.info("Initializing explainer")
    explainer = ADEExplainer(
        entity_extractor=entity_extractor,
        severity_classifier=severity_classifier
    )
    
    explanations = []
    for i, text in enumerate(text_examples):
        logger.info(f"Generating explanation for example {i+1}/{len(text_examples)}")
        explanation = explainer.generate_explanation(text)
        explanations.append(explanation)
        
    return explanations


def main():
    """Main function to demonstrate the ADE model workflow."""
    parser = argparse.ArgumentParser(description="Train and evaluate ADE models")
    parser.add_argument("--ner-annotations", type=str, help="Path to Label Studio NER annotations")
    parser.add_argument("--severity-file", type=str, help="Path to severity labels CSV")
    parser.add_argument("--symptoms-file", type=str, help="Path to symptoms CSV")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save models")
    args = parser.parse_args()
    
    # Create model directories
    ner_model_dir = os.path.join(args.model_dir, "ner")
    severity_model_dir = os.path.join(args.model_dir, "severity")
    os.makedirs(ner_model_dir, exist_ok=True)
    os.makedirs(severity_model_dir, exist_ok=True)
    
    # Train entity extractor
    entity_extractor = train_entity_extractor(ner_model_dir, args.ner_annotations)
    
    # Train severity classifier
    severity_classifier = train_severity_classifier(severity_model_dir, args.severity_file)
    
    # Cluster symptoms
    symptom_clusterer, clusters = cluster_symptoms(args.symptoms_file)
    
    # Generate explanations for example texts
    example_texts = [
        "Patient experienced severe headache and nausea after taking ibuprofen.",
        "Two days after receiving the vaccine, the patient reported mild fever and fatigue.",
        "The patient developed a severe rash and difficulty breathing after taking amoxicillin."
    ]
    
    explanations = generate_explanations(example_texts, entity_extractor, severity_classifier)
    
    # Print explanations
    for i, (text, explanation) in enumerate(zip(example_texts, explanations)):
        print(f"\n=== Example {i+1} ===")
        print(f"Text: {text}")
        
        # Print entities
        print("\nExtracted Entities:")
        for entity in explanation["entities"]:
            print(f"  - {entity['type']}: {entity['text']} ({entity['start']}:{entity['end']})")
        
        # Print severity
        if explanation["severity"]:
            print(f"\nSeverity: {explanation['severity']['severity']} (Confidence: {explanation['severity']['confidence']:.2f})")
            print("Probability Distribution:")
            for label, prob in explanation["severity"]["probabilities"].items():
                print(f"  - {label}: {prob:.2f}")
        
        # Print top features for prediction
        if explanation["severity_explanation"]:
            print("\nTop features influencing prediction:")
            for label, features in explanation["severity_explanation"]["explanations"].items():
                if label == explanation["severity"]["severity"]:
                    for feature, weight in features[:5]:
                        print(f"  - {feature}: {weight:.4f}")
    
    logger.info("ADE model workflow completed")


if __name__ == "__main__":
    main()
