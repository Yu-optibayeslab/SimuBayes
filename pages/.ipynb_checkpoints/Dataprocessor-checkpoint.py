import streamlit as st
import pandas as pd
import numpy as np
# Import your custom classes (adjust paths as needed)
from dataprocessors import DataCleaner, MissingDataHandler, DimensionalityReducer

def app():
    # Retrieve raw data from session state
    data = st.session_state.get("raw_data", None)
    processed_data = data.copy()
    
    # Initialize processing classes
    cleaner = DataCleaner()
    missing_handler = MissingDataHandler()
    reducer = DimensionalityReducer()

    st.title("Data processing")
    
    st.write("## Processing Steps ##")
    
    # Data Cleansing Options
    st.sidebar.subheader("Data Cleansing")
    outlier_method = st.sidebar.selectbox("Outlier Method", options=["iqr", "zscore"])
    if st.sidebar.button("Remove Outliers", key="unique_key_1"):
        # Perform outlier removal action here
        processed_data = cleaner.remove_outliers(processed_data, method=outlier_method)
        st.success(f"Outliers are removed using method: **{outlier_method}**")
        
    if st.sidebar.button("Removing duplicates", key="unique_key_2"):
        processed_data = cleaner.remove_duplicates(processed_data)
        st.success(f"Duplicates are removed!\n")
            
    # Missing Data Handling Options
    st.sidebar.subheader("Missing Data Handling")
    imputation_method = st.sidebar.selectbox("Imputation Method", options=["mean", "median", "simple", "knn", "mice"])
    
    if st.sidebar.button("Filling missing Data", key="unique_key_3"):
        st.write("### Missing Data Handling")
        if imputation_method == "mean":
            processed_data = missing_handler.mean_imputation(processed_data)
        elif imputation_method == "median":
            processed_data = missing_handler.median_imputation(processed_data)
        elif imputation_method == "most_frequent":
            processed_data = missing_handler.mode_imputation(processed_data)
        elif imputation_method == "knn": 
            processed_data = missing_handler.knn_imputation(processed_data)
        else:
            # The last is reserved for a mice_imputation method
            processed_data = missing_handler.mice_imputation(processed_data)
        st.success(f"Missing Data are filled using method: **{imputation_method}**")
        
    # Dimensionality Reduction Options
    st.sidebar.subheader("Dimensionality Reduction")
    #perform_reduction = st.sidebar.checkbox("Perform Dimensionality Reduction", value=True)
    reduction_method = st.sidebar.selectbox("Reduction Method", options=["pca", "tsne"])

    if st.sidebar.button("Reducing Dimensionality", key="unique_key_4"):
        st.write("### Dimensionality Reduction")
        features = processed_data.drop(columns=["target"])
        if reduction_method == "pca":
            n_components = st.sidebar.slider("Number of Components", min_value=2, max_value=5, value=2)
            reduced = reducer.pca(features, n_components=n_components)
            reduced_df = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
            reduced_df["target"] = processed_data["target"].values
            #st.write("Reduced Data:")
            #st.write(reduced_df.head())
            # Save reduced data in session state so it can be used in the visualizations page
            st.session_state["reduced_data"] = reduced_df
        elif reduction_method == "tsne":
            st.session_state["reduced_data"] = None
        else:
            st.session_state["reduced_data"] = None
        st.success(f"Missing Data are filled using method: **{imputation_method}**")
        
    # Save the processed data in session state
    st.session_state["processed_data"] = processed_data

    st.write("Processed data is now available for visualizations on the **Visualizations** page.")

    # --- Save Processed Data ---
    st.sidebar.subheader("~~ Save Processed Data ~~")
    save_format = st.sidebar.selectbox("Select file format", options=["CSV", "JSON"])
    output_dir = "./outputs"
    if st.sidebar.button("Save Data", key="unique_key_5"):
        os.makedirs(output_dir, exist_ok=True)
        if save_format == "CSV":
            csv_data = processed_data.to_csv(index=False)
            filename = os.path.join(output_dir, "processed_data.csv")
            st.sidebar.download_button("Download CSV", data=csv_data, file_name=filename, mime="text/csv")
        else:
            json_data = processed_data.to_json(orient="records")
            filename = os.path.join(output_dir, "processed_data.json")
            st.sidebar.download_button("Download JSON", data=json_data, file_name=filename, mime="application/json")
