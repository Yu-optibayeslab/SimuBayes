import streamlit as st
import pandas as pd
import numpy as np

# Helper function to load sample data
def load_sample_data():
    data = pd.DataFrame({
        "feature1": np.random.normal(10, 2, 100),
        "feature2": np.random.normal(20, 5, 100),
        "feature3": np.random.normal(30, 10, 100),
        "target": np.random.choice([0, 1], 100)
    })
    data.loc[10:15, "feature1"] = np.nan
    data.loc[20:25, "feature2"] = np.nan
    data.loc[30:35, "feature3"] = 1000
    return data

def app():
    st.title("Data Processing Pipeline")
    st.sidebar.title("Data Processor")
    #st.sidebar.header("Configuration")
    
    # Data Loading
    if st.sidebar.button("Load Sample Data", key="unique_key_0"):
        data = load_sample_data()
        st.session_state["raw_data"] = data
        st.success("Sample data loaded!")

    if "raw_data" not in st.session_state:
        st.warning("Please load the sample data from the sidebar.")
        return