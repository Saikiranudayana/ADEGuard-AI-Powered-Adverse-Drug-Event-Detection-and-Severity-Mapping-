"""
Script for performing real-time ADE detection and analysis.

This script demonstrates how to:
1. Load trained ADE models
2. Extract ADE entities from text
3. Classify ADE severity
4. Cluster similar symptoms
5. Generate explanations for predictions

Usage:
    python scripts/inference.py --text "Patient experienced severe headache after taking aspirin."
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Optional
import json

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


def load_models(model_base_dir: str) -> tuple:
    """
    Load all trained ADE models.
    
    Args:
        model_base_dir: Base directory containing trained models
        
    Returns:
        Tuple of (entity_extractor, severity_classifier, symptom_clusterer, explainer)
    """
    logger.info("Loading trained models")
    
    # Model directories
    ner_model_dir = os.path.join(model_base_dir, "ner")
    severity_model_dir = os.path.join(model_base_dir, "severity")
    
    # Check if model directories exist
    if not os.path.exists(ner_model_dir):
        logger.warning(f"NER model directory not found: {ner_model_dir}. Using default model.")
    
    if not os.path.exists(severity_model_dir):
        logger.warning(f"Severity model directory not found: {severity_model_dir}. Using default model.")
    
    # Load entity extractor
    entity_extractor = ADEEntityExtractor(model_path=ner_model_dir if os.path.exists(ner_model_dir) else None)
    
    # Load severity classifier
    severity_classifier = ADESeverityClassifier(model_path=severity_model_dir if os.path.exists(severity_model_dir) else None)
    
    # Initialize symptom clusterer
    symptom_clusterer = ADESymptomClusterer()
    
    # Initialize explainer
    explainer = ADEExplainer(
        entity_extractor=entity_extractor,
        severity_classifier=severity_classifier
    )
    
    logger.info("Models loaded successfully")
    
    return entity_extractor, severity_classifier, symptom_clusterer, explainer


def extract_entities(text: str, entity_extractor: ADEEntityExtractor) -> Dict:
    """
    Extract ADE entities from text.
    
    Args:
        text: Input text
        entity_extractor: Trained entity extractor model
        
    Returns:
        Dictionary of extracted entities
    """
    logger.info("Extracting entities")
    entities = entity_extractor.extract_entities(text)
    
    # Format entities for display
    formatted_entities = []
    for entity in entities:
        formatted_entity = {
            "text": entity["text"],
            "type": entity["type"],
            "start": entity["start"],
            "end": entity["end"],
            "confidence": entity.get("confidence", None)
        }
        formatted_entities.append(formatted_entity)
    
    return {"text": text, "entities": formatted_entities}


def classify_severity(text: str, severity_classifier: ADESeverityClassifier) -> Dict:
    """
    Classify ADE severity.
    
    Args:
        text: Input text
        severity_classifier: Trained severity classifier model
        
    Returns:
        Dictionary with severity classification results
    """
    logger.info("Classifying severity")
    severity = severity_classifier.classify_severity(text)
    
    return {
        "text": text,
        "severity": severity["severity"],
        "confidence": severity["confidence"],
        "probabilities": severity["probabilities"]
    }


def cluster_text_symptoms(text: str, entity_extractor: ADEEntityExtractor, symptom_clusterer: ADESymptomClusterer) -> Dict:
    """
    Extract symptoms from text and find their clusters.
    
    Args:
        text: Input text
        entity_extractor: Trained entity extractor model
        symptom_clusterer: Symptom clusterer
        
    Returns:
        Dictionary with symptom clusters
    """
    logger.info("Extracting and clustering symptoms")
    
    # First extract entities
    entities = entity_extractor.extract_entities(text)
    
    # Filter for symptom entities
    symptoms = [entity["text"] for entity in entities if entity["type"] == "SYMPTOM"]
    
    if not symptoms:
        logger.info("No symptoms found in text")
        return {"text": text, "symptoms": [], "clusters": {}}
    
    # Cluster symptoms
    clusters = symptom_clusterer.cluster_symptoms(symptoms=symptoms)
    
    # Format result
    result = {
        "text": text,
        "symptoms": symptoms,
        "clusters": {}
    }
    
    # Add cluster information
    if clusters and "clusters" in clusters:
        for symptom, cluster_id in zip(symptoms, clusters["clusters"]):
            if cluster_id not in result["clusters"]:
                result["clusters"][cluster_id] = []
            result["clusters"][cluster_id].append(symptom)
    
    return result


def generate_explanation(text: str, explainer: ADEExplainer) -> Dict:
    """
    Generate complete explanation for ADE text.
    
    Args:
        text: Input text
        explainer: ADE explainer
        
    Returns:
        Dictionary with complete explanation
    """
    logger.info("Generating explanation")
    explanation = explainer.generate_explanation(text)
    
    # Format for better readability
    result = {
        "text": text,
        "entities": explanation["entities"],
        "severity": explanation["severity"],
        "explanation": {}
    }
    
    # Add feature importance for each severity level
    if explanation.get("severity_explanation", None):
        result["explanation"] = explanation["severity_explanation"]
    
    return result


def process_batch(texts: List[str], models: tuple) -> List[Dict]:
    """
    Process a batch of texts with all ADE models.
    
    Args:
        texts: List of input texts
        models: Tuple of models (entity_extractor, severity_classifier, symptom_clusterer, explainer)
        
    Returns:
        List of results for each text
    """
    entity_extractor, severity_classifier, symptom_clusterer, explainer = models
    
    results = []
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i+1}/{len(texts)}")
        
        # Generate comprehensive explanation
        explanation = explainer.generate_explanation(text)
        
        # Extract symptoms for clustering
        symptoms = [entity["text"] for entity in explanation["entities"] 
                   if entity["type"] == "SYMPTOM"]
        
        # Add cluster information if symptoms are found
        if symptoms:
            cluster_info = symptom_clusterer.cluster_symptoms(symptoms=symptoms)
            
            # Map symptoms to clusters
            symptom_clusters = {}
            if "clusters" in cluster_info:
                for symptom, cluster_id in zip(symptoms, cluster_info["clusters"]):
                    if cluster_id not in symptom_clusters:
                        symptom_clusters[cluster_id] = []
                    symptom_clusters[cluster_id].append(symptom)
                
                explanation["symptom_clusters"] = symptom_clusters
        
        results.append(explanation)
    
    return results


def main():
    """Main function to demonstrate ADE inference."""
    parser = argparse.ArgumentParser(description="Perform ADE inference")
    parser.add_argument("--text", type=str, help="Input text for ADE analysis")
    parser.add_argument("--file", type=str, help="File with multiple texts (one per line)")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory with trained models")
    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    args = parser.parse_args()
    
    # Check if either text or file is provided
    if not args.text and not args.file:
        parser.error("Either --text or --file must be provided")
    
    # Load models
    models = load_models(args.model_dir)
    
    # Process input
    if args.text:
        # Process single text
        texts = [args.text]
    else:
        # Process texts from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Process all texts
    results = process_batch(texts, models)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        # Print results to console
        for i, result in enumerate(results):
            print(f"\n=== Result {i+1} ===")
            print(f"Text: {result['text']}")
            
            # Print entities
            print("\nExtracted Entities:")
            for entity in result["entities"]:
                print(f"  - {entity['type']}: {entity['text']} ({entity['start']}:{entity['end']})")
            
            # Print severity
            if result["severity"]:
                print(f"\nSeverity: {result['severity']['severity']} (Confidence: {result['severity']['confidence']:.2f})")
                print("Probability Distribution:")
                for label, prob in result["severity"]["probabilities"].items():
                    print(f"  - {label}: {prob:.2f}")
            
            # Print top features for prediction
            if "severity_explanation" in result:
                print("\nTop features influencing prediction:")
                for label, features in result["severity_explanation"]["explanations"].items():
                    if label == result["severity"]["severity"]:
                        for feature, weight in features[:5]:
                            print(f"  - {feature}: {weight:.4f}")
            
            # Print cluster information
            if "symptom_clusters" in result:
                print("\nSymptom Clusters:")
                for cluster_id, symptoms in result["symptom_clusters"].items():
                    print(f"  - Cluster {cluster_id}: {', '.join(symptoms)}")


if __name__ == "__main__":
    main()
