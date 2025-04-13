import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Import your custom classes (adjust paths as needed)
from dataprocessors import DataCleaner, MissingDataHandler, DimensionalityReducer
from dataprocessors import DataNormaliser, DimensionalityReducer
from dataprocessors import Visualiser

from streamlit.runtime.scriptrunner import get_script_run_ctx

def get_current_tab():
    ctx = get_script_run_ctx()
    if ctx is None:
        return None
    query_params = st.query_params
    return query_params.get("tab", None)

def load_csv_data(file_path):
    try:
        data = pd.read_csv(file_path, sep=None, engine='python')
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Function to log actions
def log_action(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.actions.append({"timestamp": timestamp, "action": action})


def app():
    # Add CSS to style the tabs
    st.markdown("""
    <style>
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
            "1: Raw Training Data Loader", 
            "2: Training Data Cleaner",
            "3: Training Data Transformer",
            "4: Training Data Visualisor"
            ])

    with tab1:
        st.header("SimuBayes: Data loading page.")
        # Initialize session state for working_directory
        if "working_directory" not in st.session_state:
            st.session_state["working_directory"] = []

        if "working_dir_confirmed" not in st.session_state:
            st.session_state.working_dir_confirmed = False

        if "inputs_directory" not in st.session_state:
            st.session_state["inputs_directory"] = []

        if "outputs_directory" not in st.session_state:
            st.session_state["outputs_directory"] = []

        if "raw_data" not in st.session_state:
            st.session_state["raw_data"] = pd.DataFrame()

        if "preprocessed_data" not in st.session_state:
            st.session_state["preprocessed_data"] = pd.DataFrame()
            
        DEFUALT_WORKING_DIRECTORY = os.getcwd()
        # Working Directory input
        working_directory = st.text_input("Enter Your Working Directory:", value = DEFUALT_WORKING_DIRECTORY, key="working_directory_input")
        st.write(f"Current Directory is: {DEFUALT_WORKING_DIRECTORY}")

        if st.button("Confirm working directory"):
            st.session_state.working_dir_confirmed = True

        # Store the working directory in session state
        if st.session_state.working_dir_confirmed:
            st.session_state["working_directory"] = working_directory
            os.makedirs(working_directory, exist_ok=True)
            st.success(f"Working Directory is now: {working_directory}")
                # Display the working directory
            if "working_directory" in st.session_state:
                #st.write(f"Current Working Directory: {st.session_state['working_directory']}")
                inputs_dir = os.path.join(working_directory, "inputs")
                outputs_dir = os.path.join(working_directory, "outputs")
                st.session_state["inputs_directory"] = inputs_dir
                st.session_state["outputs_directory"] = outputs_dir
                st.success(f"Please move all raw .csv data files to: {inputs_dir}! .")
                st.sidebar.warning(f"The files downloaded in the actions will be savd as the default download directoy of your webbrowser. \
                            Please move them to {outputs_dir} as they may needed in further actions! ")
                os.makedirs(inputs_dir, exist_ok=True)
                os.makedirs(outputs_dir, exist_ok=True)
                inputs_directory = st.session_state["inputs_directory"]

                # Process CSVs in the directory
                if inputs_directory and os.path.isdir(inputs_directory):
                    csv_files = [f for f in os.listdir(inputs_directory) if f.endswith(".csv")]
                    ##############################
                    # Sidebar for chunk settings #
                    ##############################
                    if csv_files:
                        # Dropdown to select a document
                        selected_csv = st.selectbox(
                            "Load raw training data file (.csv)",
                            options=csv_files,
                            index=None  # No document is selected by default
                        )
                        # Clear session state if a new CSV file is selected
                        if "selected_csv" in st.session_state and st.session_state.selected_csv != selected_csv:
                            st.session_state.selected_csv = ""  # Reset only the selected_csv key
                        st.session_state.selected_csv = selected_csv  # Update the selected CSV in session state
        
                        if st.session_state.selected_csv:
                            file_path = os.path.join(inputs_directory, selected_csv)
                            # Data Loading
                            if st.button("Load CSV Data", key="unique_key_0"):
                                data = load_csv_data(file_path)
                                if data is not None:
                                    st.session_state["raw_data"] = data
                                    st.success(f"CSV data loaded from: {file_path}!")
                                    st.session_state["preprocessed_data"] = data
                                    st.dataframe(st.session_state["raw_data"])  # Display the loaded data
                                    # Identify columns that have any NaN values
                                    nan_columns = data.columns[data.isna().any()].tolist()
                                    if nan_columns:
                                        st.write("Columns with NaN values:", nan_columns)# Display the list of columns with NaN values
                                else:
                                    st.error("Failed to load data.")
                        if "raw_data" not in st.session_state:
                            st.warning("Please load the CSV data from the sidebar.")
                            return
                    else:
                        st.error(f"No .csv files found!!! \n ### Please move your .csv files to the {inputs_dir} ###")
        else:
            st.warning("Please enter the directory!")

    with tab2:
        # Retrieve raw data from session state
        #if "raw_data" not in st.session_state:
        #    st.write()
        #else:
        data = st.session_state.get("raw_data", None)
        preprocessed_data = data.copy()
    
        # Initialize processing classes
        cleaner = DataCleaner()
        missing_handler = MissingDataHandler()
        reducer = DimensionalityReducer()

        st.title("SimuBayes: Data processing page.")
    
        st.write("## Processing Steps ##")
        num_rows_all_nan = data.isna().any(axis=1).sum() # Check the number of rows with nan values
        if num_rows_all_nan:
            st.error(f"### Missing data found. \nNumber of rows contains nan: {num_rows_all_nan}.\n \
                    Please handle the missing data using a function in the sidebar ###")# Display the number of rows with nan values

        num_duplicated_rows = data.duplicated(keep=False).sum() #Check the number of duplicated rows in a pandas DataFrame
        if num_duplicated_rows:
            st.warning(f"### Duplicated rows (including all occurrences): {num_duplicated_rows}.\n \
                    Please remove the duplicates using the function in the sidebar ###")

        # Data Cleansing Options
        st.markdown("---")  # Shortcut for <hr> in Markdown
        if st.button("Remove duplicates", key="unique_key_2"):
            preprocessed_data = cleaner.remove_duplicates(preprocessed_data)
            st.success(f"Duplicates are removed!\n")
        st.markdown("---")  # Shortcut for <hr> in Markdown
        st.subheader("Data Cleansing")
        outlier_method = st.selectbox("Outlier Method", options=["iqr", "zscore"])
        if st.button("Remove Outliers", key="unique_key_1"):
            # Perform outlier removal action here
            preprocessed_data = cleaner.remove_outliers(preprocessed_data, method=outlier_method)
            st.success(f"Outliers are removed using method: **{outlier_method}**")

        st.markdown("---")  # Shortcut for <hr> in Markdown  
        # Missing Data Handling Options
        st.subheader("Missing Data Handling")
        imputation_method = st.selectbox("Imputation Method", options=["remove nan", "mean", "median", "simple", "knn", "mice"])
    
        if st.button("Handle missing Data", key="unique_key_3"):
            st.write("### Missing Data Handling")
            if imputation_method == "mean":
                preprocessed_data = missing_handler.mean_imputation(preprocessed_data)
            elif imputation_method == "median":
                preprocessed_data = missing_handler.median_imputation(preprocessed_data)
            elif imputation_method == "most_frequent":
                preprocessed_data = missing_handler.mode_imputation(preprocessed_data)
            elif imputation_method == "knn": 
                preprocessed_data = missing_handler.knn_imputation(preprocessed_data)
            elif imputation_method == "mice": 
                preprocessed_data = missing_handler.mice_imputation(preprocessed_data)
            else:
                # The last is reserved for Removing rows with any NaN values
                preprocessed_data == preprocessed_data.dropna()
            
            st.success(f"Missing Data are filled using method: **{imputation_method}**")

        st.markdown("---")  # Shortcut for <hr> in Markdown    
        # Save the processed data in session state
        st.session_state["preprocessed_data"] = preprocessed_data

        st.write("Processed data is now available for visualizations on the **Visualizations** page.")

        # --- Save Processed Data ---
        st.subheader("~~~~~~ Save Processed Data ~~~~~~")
        save_format = st.selectbox("Select file format", options=["CSV", "JSON"])

        output_dir = st.session_state["outputs_directory"]
        if st.button("Save Data", key="unique_key_5"):
            #os.makedirs(output_dir, exist_ok=True)
            if save_format == "CSV":
                csv_data = preprocessed_data.to_csv(index=False)
                filename = os.path.join(output_dir, "preprocessed_data.csv")
                st.write(f"saving {filename}")
                preprocessed_data.to_csv(filename, index=False)
            else:
                json_data = preprocessed_data.to_json(orient="records")
                filename = os.path.join(output_dir, "preprocessed_data.json")
                st.write(f"saving {filename}")
                with open(filename, "w") as f:
                    json.dump(json_data, f, indent=4)

    with tab3:
        #Initialize st.session_state.actions if it doesn't exist
        if "actions" not in st.session_state:
            st.session_state.actions = []

        # retrive the directory for storing the outputs
        output_dir = st.session_state["outputs_directory"]

        # Retrieve raw data from session state
        processed_data = st.session_state.get("preprocessed_data", None)
    
        # Initialize processing classes
        transformer = DataNormaliser()
        reducer = DimensionalityReducer()

        st.title("SimuBayes: Data transforming page.")
        st.write("## Processing Steps ##")
    
        # Data Normalisation Options
        st.markdown("---")  # Shortcut for <hr> in Markdown   
        #st.subheader("Data normalisation")
        data_normalisation = st.checkbox("Data normalisation")
        if data_normalisation:
            #st.info("Please select the normalisation approaches in the main window!")

            # Get the list of columns
            columns = processed_data.columns.tolist()

            # Option to apply the same normalisation method to all columns
            st.subheader("Global normalisation")
            global_method = st.selectbox(
                    "Select a normalisation method to apply to ALL columns",
                    options=["none", "minmax", "standard", "robust", "l2", "log", "power"],
                    key="global_method",
                )

            # Let the user select normalisation methods for each column (if global method is "none")
            normalisation_methods = {}
            if global_method == "none":
                st.subheader("Column-wise normalisation")
                for column in columns:
                    method = st.selectbox(
                            f"Select normalisation method for column '{column}'",
                            options=["none", "minmax", "standard", "robust", "l2", "log", "power"],
                            key=column,
                        )
                    if method != "none":
                        normalisation_methods[column] = method
            else:
                # Apply the global method to all columns
                for column in columns:
                    normalisation_methods[column] = global_method
            st.write(f"normalisation_methods:\n{normalisation_methods}")

            # normalise the data if methods are selected
            if normalisation_methods:
                if st.button("normalise Data", key="normaliser"):
                    normalised_data, scalers = transformer.normalise_data(processed_data, normalisation_methods)
                    st.session_state["normalised_data"] = normalised_data
                    st.session_state["scalers"] = scalers
                    st.subheader("Normalised Data")
                    st.write(normalised_data)
                    st.subheader("Normalisation approaches")
                    st.write("Scaler applied to each input feature:")
                    for column, scaler in scalers.items():
                        st.write(f"Column: {column}, Scaler: {scaler}")
                    log_action(f"Data normalisation is applied on preprocessed raw data {normalisation_methods[column]}. \n \
                                Normalised data is stored in \'st.session_state[\"normalised_data\"]\'. \n \
                                Scalers are stored in \'st.session_state[\"scalers\"]\'.")
                    st.write(f"Actions done in the data transformer: {st.session_state.actions}!")

                    # Save scalers as pkl
                    scalers_filename = os.path.join(output_dir, "scalers_for_normalisation.pkl")
                    # Build a nested dictionary to hold both the scaler and the method for each feature:  
                    with open(scalers_filename, "wb") as f:
                        pickle.dump(scalers, f)

            # --- Save Normalised Data ---
        
            st.subheader("~~ Save Normalised Data ~~")
            save_format = st.selectbox("Select file format", options=["CSV", "JSON"], key = "normalised_data_save_format")
            if st.button("Save Normalised Data", key="save_normalised_data"):
                if "normalised_data" in st.session_state and "scalers" in st.session_state:
                    normalised_data = st.session_state["normalised_data"]
                    scalers = st.session_state["scalers"]
                    if save_format == "CSV":
                        # Save normalised data
                        file_name = os.path.join(output_dir, "normalised_data.csv")
                        st.write(f"saving {file_name}")
                        normalised_data.to_csv(file_name, index=False)
                    else:
                        # Save normalised data and scalers together in a single JSON file
                        combined_data = {
                            "normalised_data": normalised_data.to_dict(orient="records"),
                            }
                        jsonname = os.path.join(output_dir, "normalised_data_and_methods.json")
                        st.write(f"saving {jsonname}")
                        with open(jsonname, "w") as f:
                            json.dump(combined_data, f, indent=4)
                else:
                    st.warning("No normalised data found. Please normalise the data first.")
    
        ####################################
        # Dimensionality Reduction Options #
        ####################################
        st.markdown("---")  # Shortcut for <hr> in Markdown   
        st.subheader("Dimensionality Reduction")
        #perform_reduction = st.checkbox("Perform Dimensionality Reduction", value=True)
        data_to_be_reduced = st.selectbox("Which data to be reduced", \
                                                    options=["--- select an data type ---", "raw_data", "preprocessed_data", "normalised_data"])
        if data_to_be_reduced:
            st.session_state["reduced_data_which"] = data_to_be_reduced
            if data_to_be_reduced == "preprocessed_data":
                data = st.session_state.get("preprocessed_data", None)
            elif data_to_be_reduced =="normalised_data":
                data = st.session_state.get("normalised_data", None)
            else:
                data = st.session_state.get("raw_data", None)

            reduction_method = st.selectbox("Reduction Method", options=["--- select an approach ---", "pca", "tsne"])
            if reduction_method == "pca":
                columns_to_drop = st.multiselect(
                                "Select features to be excluded in the PCA analysis:",
                                options=data.columns,  # All columns are available for selection
                                default=None  # No columns selected by default
                            )
                features = data.drop(columns=columns_to_drop)
                n_components = st.text_input("Number of Components", value="2")  # Default value is 2
                n_components = int(n_components)
                if st.button("Reducing Dimensionality", key="Reducing_Dimensionality"):
                    reduced = reducer.pca(features, n_components=n_components)
                    reduced_df = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
                    reduced_df[columns_to_drop] = data[columns_to_drop].values
                    st.write("Reduced Data:")
                    st.write(reduced_df)
                    # Save reduced data in session state so it can be used in the visualizations page
                    st.session_state["reduced_data"] = reduced_df

                    log_action(f"pca is applied to reduce the dimensionality of {data_to_be_reduced}.  \
                                Considered features are {features.columns}  \
                                Normalised data is stored in \'st.session_state[\"normalised_data\"]\'.  \
                                Scalers are stored in \'st.session_state[\"scalers\"]\'.")
                    st.write(f"Actions done in the data transformer: {st.session_state.actions}!")

            elif reduction_method == "tsne":
                columns_to_drop = st.multiselect(
                                "Selected variables to be excluded in the TSNE analysis:",
                                options=data.columns,  # All columns are available for selection
                                default=None  # No columns selected by default
                            )
                features = data.drop(columns=columns_to_drop)
                st.write(features)
                n_components = st.text_input("Number of Components", value="2")  # Default value is 2
                n_components = int(n_components)
                perplexity = st.text_input("Perplexity", value="30")  # Default value is 2
                perplexity = int(perplexity)
                n_iter = st.text_input("Number of iterations (n_iter)", value="1000")  # Default value is 1000
                n_iter = int(n_iter)
        
                if st.button("Reducing Dimensionality", key="Reducing_Dimensionality"):
                    random_state = 42
                    reduced = reducer.tsne(features, n_components=n_components, perplexity=perplexity, random_state=random_state)
                    reduced_df = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
                    reduced_df[columns_to_drop] = data[columns_to_drop].values
                    st.write("Reduced Data:")
                    st.write(reduced_df)
                    # Save reduced data in session state so it can be used in the visualizations page
                    st.session_state["reduced_data"] = reduced_df
                    log_action(f"tsne is applied to reduce the dimensionality of {data_to_be_reduced}.  \
                                Considered features are {features.columns}  \
                                Normalised data is stored in \'st.session_state[\"normalised_data\"]\'.  \
                                Scalers are stored in \'st.session_state[\"scalers\"]\'.")
                    st.write(f"Actions done in the data transformer: {st.session_state.actions}!")
            else:
                if st.button("Reducing Dimensionality", key="Reducing_Dimensionality"):
                    st.error("No method selected! Please select a dimensionality reducing approach from the droplist!!!")
    
    ######################################################
    # Visualisor
    ######################################################
    with tab4:
        st.header("Visualise the raw/processed training data here!")

        st.markdown("---")  # Shortcut for <hr> in Markdown 
        st.header("Visualisation Options")

        # Initialize Visualiser instance
        visualiser = Visualiser()

        # Dropdown to select dataset
        dataset_options = ["Preprocessed Data", "Normalised Data", "Reduced Data"]
        selected_dataset = st.selectbox("Select Dataset", dataset_options)

        # Display appropriate plot based on selected dataset and plot type
        if selected_dataset == "Preprocessed Data" or selected_dataset == "Normalised Data":
            # Retrieve data from session state
            if selected_dataset == "Preprocessed Data":
                data = st.session_state.get("preprocessed_data", None)
                st.write(data)
            else:
                data = st.session_state.get("normalised_data", None)
                st.write(data)

            if data is None:
                st.warning("No valida data found. Please process data on the main page first.")
                st.stop()
            # Allow user to choose a plot type
            plot_options = [
                "---Select a plot ---",
                "Scatterplot Matrix",
                "Parallel Coordinates Plot",
                "RadViz plot",
                "Heatmap (Correlation Matrix)"
                ]
            selected_plot = st.selectbox("Select a plot", plot_options)
            if selected_plot == "Scatterplot Matrix":
                st.subheader("Scatterplot Matrix")
                hue = st.selectbox(
                                            "hue (feature name for coloring):",
                                            options=data.columns,  # All columns are available for selection
                                            key = "selectbox_hue"
                                            )
                st.write(f"hue = {hue}!\n")
                fig = visualiser.scatterplot_matrix(data, hue=hue)
                st.pyplot(fig)

            elif selected_plot == "Parallel Coordinates Plot":
                st.subheader("Parallel Coordinates Plot")
                class_column = st.selectbox(
                                                    "class_column:",
                                                    options=data.columns,  # All columns are available for selection
                                                    key = "selectbox_class_column1"
                                                    )
                fig = visualiser.parallel_coordinates_plot(data, class_column=class_column)
                st.pyplot(fig)

            elif selected_plot == "RadViz plot":
                st.subheader("RadViz plot")
                class_column = st.selectbox(
                                                    "class_column:",
                                                    options=data.columns,  # All columns are available for selection
                                                    key = "selectbox_class_column1"
                                                    )
                fig = visualiser.radviz(data, class_column=class_column)
                st.pyplot(fig)

            elif selected_plot == "Heatmap (Correlation Matrix)":
                st.subheader("Heatmap (Correlation Matrix)")
                fig = visualiser.heatmap(data.corr())
                st.pyplot(fig)

            else:
                st.warning("Please select a type of plot to present.")
        elif selected_dataset == "Reduced Data":
            reduced_data = st.session_state.get("reduced_data", None)
            data = st.session_state.get("preprocessed_data", None)
            if reduced_data is None:
                st.warning("No processed data found. Please process data on the main page first.")
                st.stop()
            # Allow user to choose a plot type
            plot_options = [
                "---Select a plot ---",
                "2D Scatter Plot (Reduced Data)",
                "3D Scatter Plot (Reduced Data)",
                ]
            selected_plot = st.selectbox("Select a plot", plot_options)
            if reduced_data is not None:
                if selected_plot == "3D Scatter Plot (Reduced Data)":
                    st.subheader("3D Scatter Plot of Reduced Data")
                    x = st.selectbox(
                                        "x-axis:",
                                        options=reduced_data.columns,  # All columns are available for selection
                                        key = "selectbox_PC1"
                                    )

                    y = st.selectbox(
                                        "y-axis:",
                                        options=reduced_data.columns,  # All columns are available for selection
                                        key = "selectbox_PC2"
                                    )

                    z = st.selectbox(
                                        "z-axis:",
                                        options=reduced_data.columns,  # All columns are available for selection
                                        key = "selectbox_z"
                                    )
                    fig = visualiser.scatter_3d(reduced_data, x=x, y=y, z=z, hue=z)
                    #st.pyplot(fig)
                    st.plotly_chart(fig, use_container_width=True)
                elif selected_plot == "2D Scatter Plot (Reduced Data)":
                    st.subheader("2D Scatter Plot of Reduced Data")
                    x = st.selectbox(
                                        "x-axis:",
                                        options=reduced_data.columns,  # All columns are available for selection
                                        key = "selectbox_PC1"
                                    )

                    y = st.selectbox(
                                        "y-axis:",
                                        options=reduced_data.columns,  # All columns are available for selection
                                        key = "selectbox_PC2"
                                    )

                    z = st.selectbox(
                                        "Output:",
                                        options=reduced_data.columns,  # All columns are available for selection
                                        key = "selectbox_z"
                                    )
                    fig = visualiser.scatter_2d(reduced_data, x=x, y=y, z=z, s=20)
                    st.pyplot(fig)
            else:
                st.info("Reduced data is not available. Please perform dimensionality reduction on the main page.")


        # Self defined code for visulisation
        st.markdown("---")  # Shortcut for <hr> in Markdown 
        # Sidebar checkbox to enable/disable code editing
        enable_editing = st.checkbox("Enable User Defined Code")
        # Default code (can be predefined or loaded from a file)
        default_code = """
# Please enter your own visualisation code here.
# The following is the example code:
df = st.session_state.get("preprocessed_data", None)

# List of columns to plot
columns_to_plot = df.columns  # Use all columns in the DataFrame

# Determine the number of rows and columns for the subplot grid
num_plots = len(columns_to_plot)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

# Create a figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Loop through the columns and create subplots
for i, column in enumerate(columns_to_plot):
    ax = axes[i]
    sns.violinplot(y=df[column], ax=ax)  # Use a violin plot for each column
    ax.set_title(f'Violin Plot of {column}')
    ax.set_ylabel(column)  # Add the column name as the y-axis label

# Hide any unused subplots
for j in range(i + 1, num_rows * num_cols):
    axes[j].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the figure in Streamlit by hitting the 'Generate Plot' button.
"""

        # Display the code editor if editing is enabled
        if enable_editing:
            st.warning(" #### Please enter your own code for data visualisation below. ####")
            st.warning(" Current version support four libraries:\n 1. streamlit as st;\n 2. matplotlib.pyplot as plt; \
                        \n 3. seaborn as sns;\n 4. plotly.express as px.\n")
            st.warning("The \'raw_data\', \'preprocessed_data\', \'normalised_data\', and demensionally \'reduced_data\' are stored in: st.session_state\n \
                        you can retrieve them simply using \n\n \
                        data = st.session_state.get(\"raw_data\", None)")
            user_code = st.text_area("Edit your Python code here:", value=default_code, height=300)
            # Button to execute the code
            if st.button("Generate Plot", key="generate_plot"):
                try:
                    # Define a safe execution environment
                    allowed_globals = {
                        "plt": plt,
                        "sns": sns,
                        "px": px,
                        "st": st
                    }
                    allowed_locals = {}

                    # Execute the user's code in a restricted environment
                    exec(user_code, allowed_globals, allowed_locals)

                    # Display the plot if it was created
                    if plt.gcf().get_axes():
                        st.pyplot(plt.gcf())
                    else:
                        st.warning("No plot was generated. Did you use plt, sns, or px?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clear the plot to avoid overlapping with future plots
                    plt.clf()