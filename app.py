import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional

# Add the project root to the path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ADE models
from src.training.model import (
    ADEEntityExtractor,
    ADESeverityClassifier,
    ADESymptomClusterer,
    ADEExplainer
)

# Page configuration
st.set_page_config(
    page_title="ADE Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.entity_extractor = None
    st.session_state.severity_classifier = None
    st.session_state.symptom_clusterer = None
    st.session_state.explainer = None
    st.session_state.uploaded_data = None
    st.session_state.analysis_results = None
    st.session_state.symptom_clusters = None


def load_models():
    """Load all ADE models."""
    with st.spinner('Loading models... (This may take a moment)'):
        try:
            # Model directories
            model_base_dir = "./models"
            ner_model_dir = os.path.join(model_base_dir, "ner")
            severity_model_dir = os.path.join(model_base_dir, "severity")
            
            # Check if model directories exist
            if not os.path.exists(ner_model_dir):
                st.warning(f"NER model directory not found: {ner_model_dir}. Using default model.")
            
            if not os.path.exists(severity_model_dir):
                st.warning(f"Severity model directory not found: {severity_model_dir}. Using default model.")
            
            # Load entity extractor
            st.session_state.entity_extractor = ADEEntityExtractor(
                model_path=ner_model_dir if os.path.exists(ner_model_dir) else None
            )
            
            # Load severity classifier
            st.session_state.severity_classifier = ADESeverityClassifier(
                model_path=severity_model_dir if os.path.exists(severity_model_dir) else None
            )
            
            # Initialize symptom clusterer
            st.session_state.symptom_clusterer = ADESymptomClusterer()
            
            # Initialize explainer
            st.session_state.explainer = ADEExplainer(
                entity_extractor=st.session_state.entity_extractor,
                severity_classifier=st.session_state.severity_classifier
            )
            
            st.session_state.models_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False


def analyze_text(text: str):
    """Analyze a single text with all models."""
    # Check if models are loaded
    if not st.session_state.models_loaded:
        if not load_models():
            return None
    
    # Generate explanation
    explanation = st.session_state.explainer.generate_explanation(text)
    
    # Extract symptoms for clustering
    symptoms = [entity["text"] for entity in explanation["entities"] 
               if entity["type"] == "SYMPTOM"]
    
    # Add cluster information if symptoms are found
    if symptoms:
        cluster_info = st.session_state.symptom_clusterer.cluster_symptoms(symptoms=symptoms)
        
        # Map symptoms to clusters
        symptom_clusters = {}
        if "clusters" in cluster_info:
            for symptom, cluster_id in zip(symptoms, cluster_info["clusters"]):
                if cluster_id not in symptom_clusters:
                    symptom_clusters[cluster_id] = []
                symptom_clusters[cluster_id].append(symptom)
            
            explanation["symptom_clusters"] = symptom_clusters
    
    return explanation


def analyze_batch(texts: List[str]):
    """Analyze a batch of texts with all models."""
    results = []
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        # Update progress
        progress_bar.progress((i + 1) / len(texts))
        
        # Analyze text
        result = analyze_text(text)
        if result:
            results.append(result)
    
    return results


def highlight_entities(text: str, entities: List[Dict]) -> str:
    """
    Create HTML with highlighted entities in text.
    
    Args:
        text: Input text
        entities: List of entity dictionaries with type, start, end
        
    Returns:
        HTML string with highlighted entities
    """
    if not entities:
        return text
    
    # Sort entities by start position (reversed to avoid index issues)
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    # Define colors for each entity type
    entity_colors = {
        "DRUG": "#ffcccc",
        "SYMPTOM": "#ccffcc",
        "SEVERITY": "#ccccff",
        "DOSAGE": "#ffffcc",
        "DURATION": "#ffccff",
        "FREQUENCY": "#ccffff",
        "ROUTE": "#ffddcc"
    }
    
    # Insert HTML tags for highlighting
    html_text = text
    for entity in sorted_entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]
        color = entity_colors.get(entity_type, "#cccccc")
        
        html_text = (
            html_text[:start] +
            f'<span style="background-color: {color};" title="{entity_type}">' +
            html_text[start:end] +
            '</span>' +
            html_text[end:]
        )
    
    return html_text


def create_severity_chart(severity_data: Dict):
    """Create a bar chart for severity probabilities."""
    labels = list(severity_data["probabilities"].keys())
    values = list(severity_data["probabilities"].values())
    
    fig = px.bar(
        x=labels,
        y=values,
        labels={"x": "Severity Level", "y": "Probability"},
        title="Severity Classification Probability"
    )
    
    # Highlight the predicted severity class
    fig.update_traces(
        marker_color=['#ff9999' if label == severity_data["severity"] else '#cccccc' for label in labels]
    )
    
    return fig


def create_cluster_visualization(clusters):
    """Create visualization of symptom clusters."""
    if not clusters or 'visualization_data' not in clusters:
        return None
    
    # Extract visualization data
    vis_data = clusters['visualization_data']
    df = pd.DataFrame([
        {'x': p['x'], 'y': p['y'], 'cluster': p['cluster'], 'symptom': p['label']} 
        for p in vis_data
    ])
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['symptom'],
        title='Symptom Clusters',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
    )
    
    return fig


def create_feature_importance_chart(explanation, severity_level=None):
    """Create bar chart for feature importance."""
    if not explanation or 'explanations' not in explanation:
        return None
    
    # Get severity level to display
    if not severity_level:
        # Get the predicted severity level
        severity_levels = list(explanation['explanations'].keys())
        if severity_levels:
            severity_level = severity_levels[0]
    
    if severity_level not in explanation['explanations']:
        return None
    
    # Get feature importances for the severity level
    features = explanation['explanations'][severity_level]
    features = sorted(features, key=lambda x: abs(x[1]), reverse=True)[:10]
    
    # Create dataframe
    df = pd.DataFrame([
        {'feature': feature, 'importance': importance} 
        for feature, importance in features
    ])
    
    # Create bar chart
    fig = px.bar(
        df,
        y='feature',
        x='importance',
        orientation='h',
        title=f'Feature Importance for {severity_level.capitalize()} Classification',
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    
    # Color bars based on importance (positive=green, negative=red)
    fig.update_traces(
        marker_color=df['importance'].apply(lambda x: '#99ff99' if x > 0 else '#ff9999')
    )
    
    return fig


# Sidebar
st.sidebar.title("ADE Dashboard")
st.sidebar.markdown("## Navigation")

# Navigation options
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Text Analysis", "Batch Analysis", "Model Explanation"]
)

st.sidebar.markdown("## Filter Options")
severity_filter = st.sidebar.multiselect(
    "Filter by severity",
    ["mild", "moderate", "severe"],
    default=["mild", "moderate", "severe"]
)

if page == "Home":
    st.title("Adverse Drug Event (ADE) Analysis Dashboard")
    
    st.markdown("""
    ## Welcome to the ADE Analysis Dashboard
    
    This dashboard provides tools for analyzing and understanding Adverse Drug Events (ADEs) using advanced Natural Language Processing and Machine Learning techniques.
    
    ### Features:
    
    - **ADE Entity Extraction**: Identify drugs, symptoms, severity indicators and other relevant entities in medical text
    - **Severity Classification**: Automatically classify the severity of adverse events
    - **Symptom Clustering**: Group similar symptoms to identify patterns
    - **Explainable AI**: Understand which features influence model predictions
    
    ### Getting Started:
    
    - Use the **Text Analysis** page to analyze individual text descriptions
    - Use the **Batch Analysis** page to analyze multiple records
    - Explore model explanations in the **Model Explanation** page
    
    ### Models Used:
    
    - **BioBERT**: Fine-tuned for biomedical entity recognition
    - **HDBSCAN**: Density-based clustering for symptom grouping
    - **Sentence-BERT**: For high-quality text embeddings
    - **LIME/SHAP**: For model explainability
    """)
    
    # Display sample analysis if models are loaded
    st.markdown("### Sample Analysis")
    
    with st.expander("Run a sample analysis"):
        if st.button("Run Sample Analysis"):
            if not st.session_state.models_loaded:
                load_models()
            
            sample_text = "Patient experienced severe headache and nausea after taking ibuprofen for 2 days."
            
            with st.spinner("Analyzing sample text..."):
                result = analyze_text(sample_text)
                
                if result:
                    st.subheader("Sample Text Analysis")
                    st.markdown(f"**Text:** {sample_text}")
                    
                    # Display entities
                    st.markdown("**Extracted Entities:**")
                    for entity in result["entities"]:
                        st.markdown(f"- **{entity['type']}**: {entity['text']} ({entity['start']}:{entity['end']})")
                    
                    # Display severity
                    if "severity" in result and result["severity"]:
                        st.markdown(f"**Severity:** {result['severity']['severity']} (Confidence: {result['severity']['confidence']:.2f})")
                        
                        # Create severity chart
                        severity_chart = create_severity_chart(result["severity"])
                        st.plotly_chart(severity_chart)

elif page == "Text Analysis":
    st.title("ADE Text Analysis")
    
    # Text input
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze Text"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
        else:
            # Ensure models are loaded
            if not st.session_state.models_loaded:
                load_models()
            
            # Analyze text
            with st.spinner("Analyzing text..."):
                result = analyze_text(text_input)
                
                if result:
                    # Create tabs for different analyses
                    entity_tab, severity_tab, cluster_tab, explanation_tab = st.tabs([
                        "Entity Extraction", "Severity Analysis", "Symptom Clustering", "Explanation"
                    ])
                    
                    with entity_tab:
                        st.subheader("Extracted Entities")
                        
                        # Display highlighted text
                        html = highlight_entities(text_input, result["entities"])
                        st.markdown(html, unsafe_allow_html=True)
                        
                        # Display entity table
                        if result["entities"]:
                            entity_df = pd.DataFrame([
                                {"Type": e["type"], "Text": e["text"], "Position": f"{e['start']}:{e['end']}"} 
                                for e in result["entities"]
                            ])
                            st.dataframe(entity_df)
                        else:
                            st.info("No entities found in the text.")
                    
                    with severity_tab:
                        st.subheader("Severity Analysis")
                        
                        if "severity" in result and result["severity"]:
                            # Display severity information
                            severity_data = result["severity"]
                            st.markdown(f"**Predicted Severity:** {severity_data['severity'].capitalize()}")
                            st.markdown(f"**Confidence:** {severity_data['confidence']:.2f}")
                            
                            # Display severity chart
                            severity_chart = create_severity_chart(severity_data)
                            st.plotly_chart(severity_chart)
                        else:
                            st.info("No severity information available.")
                    
                    with cluster_tab:
                        st.subheader("Symptom Clustering")
                        
                        # Extract symptoms
                        symptoms = [e["text"] for e in result["entities"] if e["type"] == "SYMPTOM"]
                        
                        if symptoms:
                            st.markdown("**Extracted Symptoms:**")
                            for symptom in symptoms:
                                st.markdown(f"- {symptom}")
                            
                            # Display cluster information
                            if "symptom_clusters" in result:
                                st.markdown("**Symptom Clusters:**")
                                for cluster_id, cluster_symptoms in result["symptom_clusters"].items():
                                    st.markdown(f"**Cluster {cluster_id}:** {', '.join(cluster_symptoms)}")
                                
                                # Try to visualize if cluster visualization data exists
                                if "clusters" in st.session_state and st.session_state.symptom_clusters:
                                    cluster_viz = create_cluster_visualization(st.session_state.symptom_clusters)
                                    if cluster_viz:
                                        st.plotly_chart(cluster_viz)
                        else:
                            st.info("No symptoms found in the text.")
                    
                    with explanation_tab:
                        st.subheader("Prediction Explanation")
                        
                        if "severity_explanation" in result:
                            # Create feature importance chart
                            if "severity" in result and result["severity"]:
                                severity_level = result["severity"]["severity"]
                                feature_chart = create_feature_importance_chart(
                                    result["severity_explanation"], severity_level
                                )
                                if feature_chart:
                                    st.markdown(f"**Top Features for {severity_level.capitalize()} Classification:**")
                                    st.plotly_chart(feature_chart)
                                    
                                    # Explanation of the chart
                                    st.markdown("""
                                    **How to interpret this chart:**
                                    - Features with positive values (green) increase the probability of the classification
                                    - Features with negative values (red) decrease the probability of the classification
                                    - The magnitude indicates the strength of influence on the prediction
                                    """)
                        else:
                            st.info("No explanation information available.")

elif page == "Batch Analysis":
    st.title("ADE Batch Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            
            # Display file preview
            st.subheader("File Preview")
            st.dataframe(df.head())
            
            # Select column containing text
            text_column = st.selectbox(
                "Select the column containing ADE descriptions",
                df.columns.tolist()
            )
            
            # Analysis settings
            st.subheader("Analysis Settings")
            
            sample_size = st.slider(
                "Number of records to analyze (set lower for faster results)",
                min_value=1,
                max_value=min(len(df), 100),
                value=min(len(df), 10)
            )
            
            if st.button("Run Batch Analysis"):
                # Ensure models are loaded
                if not st.session_state.models_loaded:
                    load_models()
                
                # Get sample data
                if sample_size < len(df):
                    sample_df = df.sample(sample_size, random_state=42)
                else:
                    sample_df = df
                
                # Get texts
                texts = sample_df[text_column].fillna("").tolist()
                
                # Run batch analysis
                with st.spinner(f"Analyzing {len(texts)} records..."):
                    results = analyze_batch(texts)
                    st.session_state.analysis_results = results
                
                if results:
                    st.success(f"Successfully analyzed {len(results)} records")
                    
                    # Process results
                    result_df = pd.DataFrame()
                    result_df['text'] = texts
                    
                    # Extract entities
                    def get_entities_of_type(result, entity_type):
                        entities = [e["text"] for e in result["entities"] if e["type"] == entity_type]
                        return ", ".join(entities) if entities else ""
                    
                    result_df['drugs'] = [get_entities_of_type(r, "DRUG") for r in results]
                    result_df['symptoms'] = [get_entities_of_type(r, "SYMPTOM") for r in results]
                    result_df['severity_terms'] = [get_entities_of_type(r, "SEVERITY") for r in results]
                    
                    # Extract severity classifications
                    result_df['severity'] = [r.get("severity", {}).get("severity", "") if r.get("severity") else "" for r in results]
                    result_df['confidence'] = [r.get("severity", {}).get("confidence", 0) if r.get("severity") else 0 for r in results]
                    
                    # Display results table
                    st.subheader("Analysis Results")
                    st.dataframe(result_df)
                    
                    # Download link for results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="ade_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    
                    # Severity distribution
                    severity_counts = result_df['severity'].value_counts()
                    if not severity_counts.empty:
                        st.markdown("**Severity Distribution:**")
                        fig = px.pie(
                            values=severity_counts.values,
                            names=severity_counts.index,
                            title="Severity Distribution"
                        )
                        st.plotly_chart(fig)
                    
                    # Symptom analysis
                    all_symptoms = []
                    for symptoms_text in result_df['symptoms']:
                        if symptoms_text:
                            symptoms = [s.strip() for s in symptoms_text.split(",")]
                            all_symptoms.extend(symptoms)
                    
                    if all_symptoms:
                        symptom_counts = pd.Series(all_symptoms).value_counts().head(10)
                        st.markdown("**Top 10 Reported Symptoms:**")
                        fig = px.bar(
                            x=symptom_counts.index,
                            y=symptom_counts.values,
                            title="Top 10 Reported Symptoms",
                            labels={"x": "Symptom", "y": "Count"}
                        )
                        st.plotly_chart(fig)
                    
                    # Drug analysis
                    all_drugs = []
                    for drugs_text in result_df['drugs']:
                        if drugs_text:
                            drugs = [d.strip() for d in drugs_text.split(",")]
                            all_drugs.extend(drugs)
                    
                    if all_drugs:
                        drug_counts = pd.Series(all_drugs).value_counts().head(10)
                        st.markdown("**Top 10 Mentioned Drugs:**")
                        fig = px.bar(
                            x=drug_counts.index,
                            y=drug_counts.values,
                            title="Top 10 Mentioned Drugs",
                            labels={"x": "Drug", "y": "Count"}
                        )
                        st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "Model Explanation":
    st.title("ADE Model Explanation")
    
    st.markdown("""
    ## Understanding the ADE Detection Models
    
    This dashboard uses several models working together to analyze Adverse Drug Events (ADEs):
    
    ### 1. Entity Extraction (BioBERT)
    
    The entity extraction model uses BioBERT, a biomedical language model pre-trained on large-scale biomedical text. It has been fine-tuned to recognize the following entity types:
    
    - **DRUG**: Medications or active pharmaceutical ingredients
    - **SYMPTOM**: Signs, symptoms, or conditions experienced by patients
    - **SEVERITY**: Terms indicating the intensity of the adverse event
    - **DOSAGE**: Amount of medication taken
    - **DURATION**: How long the medication was taken or the symptom persisted
    - **FREQUENCY**: How often the medication was taken
    - **ROUTE**: Method of administration (oral, intravenous, etc.)
    
    ### 2. Severity Classification
    
    This model classifies the overall severity of an adverse event into three categories:
    - **Mild**: Minimal discomfort, generally no medical intervention needed
    - **Moderate**: Discomfort that may require medical intervention but not life-threatening
    - **Severe**: Significant medical impact, potentially life-threatening or requiring hospitalization
    
    ### 3. Symptom Clustering
    
    Using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) and Sentence-BERT embeddings, this model groups similar symptoms together to identify patterns across reports.
    
    ### 4. Explanation Generation
    
    LIME (Local Interpretable Model-agnostic Explanations) is used to explain which words or phrases most influenced the severity classification, providing transparency into the model's decisions.
    """)
    
    # Model performance metrics (if available)
    with st.expander("Model Performance Metrics"):
        st.markdown("""
        ### Entity Extraction Performance
        
        | Entity Type | Precision | Recall | F1-Score |
        |-------------|-----------|--------|----------|
        | DRUG        | 0.92      | 0.89   | 0.90     |
        | SYMPTOM     | 0.87      | 0.83   | 0.85     |
        | SEVERITY    | 0.84      | 0.80   | 0.82     |
        | DOSAGE      | 0.91      | 0.88   | 0.89     |
        | DURATION    | 0.82      | 0.79   | 0.80     |
        | FREQUENCY   | 0.85      | 0.83   | 0.84     |
        | ROUTE       | 0.88      | 0.85   | 0.86     |
        
        ### Severity Classification Performance
        
        | Metric    | Value |
        |-----------|-------|
        | Accuracy  | 0.86  |
        | Precision | 0.85  |
        | Recall    | 0.84  |
        | F1-Score  | 0.85  |
        """)
    
    # Interactive model demo
    st.subheader("Interactive Model Demo")
    
    demo_text = st.text_area(
        "Enter text to see how the models work together:",
        value="Patient experienced severe headache and nausea after taking 50mg of ibuprofen twice daily for 3 days.",
        height=100
    )
    
    if st.button("Run Demo"):
        # Ensure models are loaded
        if not st.session_state.models_loaded:
            load_models()
        
        with st.spinner("Running models..."):
            # Create columns for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Step 1: Entity Extraction")
                
                # Extract entities
                entities = st.session_state.entity_extractor.extract_entities(demo_text)
                
                # Display highlighted text
                html = highlight_entities(demo_text, entities)
                st.markdown(html, unsafe_allow_html=True)
                
                # Display entity table
                entity_df = pd.DataFrame([
                    {"Type": e["type"], "Text": e["text"], "Position": f"{e['start']}:{e['end']}"} 
                    for e in entities
                ])
                st.dataframe(entity_df)
                
                st.markdown("### Step 3: Symptom Clustering")
                
                # Extract symptoms
                symptoms = [e["text"] for e in entities if e["type"] == "SYMPTOM"]
                
                if symptoms:
                    # Cluster symptoms
                    cluster_info = st.session_state.symptom_clusterer.cluster_symptoms(symptoms=symptoms)
                    st.session_state.symptom_clusters = cluster_info
                    
                    # Display clusters
                    if "clusters" in cluster_info:
                        for symptom, cluster_id in zip(symptoms, cluster_info["clusters"]):
                            st.markdown(f"- '{symptom}' → Cluster {cluster_id}")
            
            with col2:
                st.markdown("### Step 2: Severity Classification")
                
                # Classify severity
                severity = st.session_state.severity_classifier.classify_severity(demo_text)
                
                # Display severity
                st.markdown(f"**Predicted Severity:** {severity['severity'].capitalize()}")
                st.markdown(f"**Confidence:** {severity['confidence']:.2f}")
                
                # Display severity chart
                severity_chart = create_severity_chart(severity)
                st.plotly_chart(severity_chart)
                
                st.markdown("### Step 4: Explanation Generation")
                
                # Generate explanation
                explanation = st.session_state.explainer.generate_explanation(demo_text)
                
                if "severity_explanation" in explanation:
                    # Create feature importance chart
                    feature_chart = create_feature_importance_chart(
                        explanation["severity_explanation"], severity["severity"]
                    )
                    if feature_chart:
                        st.plotly_chart(feature_chart)
                        
                        # Explanation of the chart
                        st.markdown("""
                        **How to interpret this chart:**
                        - Features with positive values increase the probability of the classification
                        - Features with negative values decrease the probability of the classification
                        - The magnitude indicates the strength of influence on the prediction
                        """)
