import streamlit as st

st.sidebar.title("ADE Dashboard")
st.sidebar.markdown("## Filter Options")

# Page configuration
st.set_page_config(
    page_title="ADE Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Adverse Drug Event (ADE) Analysis Dashboard")
