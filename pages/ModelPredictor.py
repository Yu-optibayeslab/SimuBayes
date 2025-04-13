import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import torch
import io
from dataprocessors import DataNormaliser, forward_uq, sensitivity_analysis
from dataprocessors import visualise_forward_uq, visualise_sensitivity_analysis, visualise_predictions
from bayesmodels import DeepGPModeler, TensorDataset, DataLoader

def app():
    # App Title
    st.title("SimuBayes: Model Predictor")

    # Initialise session state variables
    if "predicted_mean" not in st.session_state:
        st.session_state.update({
            "predicted_mean": [],
            "predicted_var": [],
            "predicted_original_mean": [],
            "predicted_original_var": [],
            "unknowninput_predictedmeanvar": None,
            "ml_config": {},
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"{torch.cuda.is_available() }")
    st.write(f"device = {device}")

    # Initial setup
    if "prev_input_columns" not in st.session_state:
        st.session_state.prev_input_columns = []
    # Define rbuttons and set a flag in session_state
    if "run_predictions" not in st.session_state:
        st.session_state.run_predictions = False

    if "input_confirmed" not in st.session_state:
        st.session_state.input_confirmed = False
        
    if "submit_output_normalisation" not in st.session_state:
        st.session_state.submit_output_normalisation = False

    if "raw_unknown_data" not in st.session_state:
        st.session_state["raw_unknown_data"] = pd.DataFrame()

    if "raw_unknown_scaled_data" not in st.session_state:
        st.session_state["raw_unknown_scaled_data"] = pd.DataFrame()

    if "input_priors_confirmed" not in st.session_state:
        st.session_state.input_priors_confirmed = False

    if "output_normalisation_method" not in st.session_state:
        st.session_state.output_normalisation_method = None

    if "visualise_forward_uq" not in st.session_state:
        st.session_state.visualise_forward_uq = False

    tab1, tab2, tab3, tab4 = st.tabs(["1: Data Loading :books:", "2: Trained Model Loading :robot_face:", \
                                            "3-1: Prediction :clipboard:", "3-2: Forward UQ & Sensitivity Analysis :bar_chart:"])

    #######################################################
    # Step 1: Data Upload and Normalisation
    #######################################################
    with tab1:
        st.header("Step 1: Upload Data and processing")
        uploaded_unknown_points = st.file_uploader("Upload CSV file containing input features", type=["csv"])

        # Check if normalisation is needed
        normalised_outputs = st.checkbox("Normalisation similar to the training needed for the prediction?", key="normalised_outputs")
        normaliser = None

        if normalised_outputs:
            saved_scalers_file = st.file_uploader("Upload file of the saved scaler", type=["pkl"])
            st.sidebar.warning("If you applied normalisation to data before, "
                       "the default save path of the .pkl file is in the 'output' directory of your working directory!")
            if saved_scalers_file:
                try:
                    saved_scalers = pickle.load(saved_scalers_file)
                    # Verify the scalers
                    for column, scaler in saved_scalers.items():
                        st.write(f"Feature: {column}, Scaler: {scaler}")
                    normaliser = DataNormaliser()
                    normaliser.scalers = saved_scalers
                    st.success("Scalers uploaded successfully!")
                except Exception as e:
                    st.error(f"Error loading scalers: {e}")

        if uploaded_unknown_points is not None:
            unknown_points_df = pd.read_csv(uploaded_unknown_points, sep=None, engine='python')
            st.session_state["raw_unknown_data"] = unknown_points_df.copy()
            st.write("The loaded data:")
            st.dataframe(st.session_state["raw_unknown_data"]) # show raw data

            if normalised_outputs and normaliser:
                unknown_points_normalised = normaliser.apply_scalers(unknown_points_df)
                unknown_points_df.iloc[:, :] = unknown_points_normalised
                st.sidebar.success("Raw data normalised using scalers from training!")
                st.write("The nomalised loaded data:")
                st.dataframe(unknown_points_df) # show normalised data
                st.session_state["raw_unknown_scaled_data"] = unknown_points_df
            st.session_state["unknowninput_predictedmeanvar"] = []
            st.success("Data loaded successfully! Please move to \"2: Trained model loading :robot_face:\"")
        else:
            st.warning("Please upload a CSV file for input features of unknown datapoints.")

    ##############################################
    # Step 2: Model Configuration
    ##############################################
    with tab2:
        st.header("Step 2: Load and Configure Model")
        options = ["--- select an approach ---", "deepGP"]
        data_modelling_method = st.selectbox("Modelling Approach of Trained Model", options)

        if data_modelling_method == "GP":
            st.warning("GP approach is under development!")
        elif data_modelling_method == "deepGP":
            # Load model configuration file
            if "ml_config" not in st.session_state:
                st.session_state["ml_config"] = {}

            st.write("Load the model configuration file 'ml_config.json'!")

            # Option to upload a JSON configuration file
            enable_uploading_config = st.checkbox("Upload ML configuration file (.json)")
            if enable_uploading_config:
                ml_config_file = st.file_uploader("Upload ml_config.json", type=["json"])
                if ml_config_file is not None:
                    try:
                        ml_config = json.load(ml_config_file)
                        st.session_state["ml_config"] = ml_config
                        st.success("Configuration file loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading JSON file: {e}")
                else:
                    st.warning("Please upload a .json file.")

            # Option to manually enter model configurations
            enable_editing = st.checkbox("Or enter model configurations")
            if enable_editing:
                input_dims = st.number_input("Input Dimension", min_value=1, value=1)
                hidden_dims = st.text_input("Hidden Dimensions List (comma-separated)", "2, 2")
                num_inducing_per_layer = st.text_input("Number of inducing points per hidden layer (comma-separated)", "64, 64, 64")
                final_outputs_dim = st.number_input("Final outputs Dimension", min_value=1, value=1)
                mean_type = st.selectbox("Mean Type", ["zero", "constant"])

                # Convert hidden_dims to list of integers
                hidden_dims = [int(x) for x in hidden_dims.split(",")]
                num_inducing_per_layer = [int(x) for x in num_inducing_per_layer.split(",")]

                # Create the ml_config dictionary
                ml_config = {
                    "data_modelling_method": data_modelling_method,
                    "input_dims": input_dims,
                    "hidden_dims": hidden_dims,
                    "num_inducing_per_layer": num_inducing_per_layer,
                    "final_outputs_dim": final_outputs_dim,
                    "mean_type": mean_type,
                }
                st.session_state["ml_config"] = ml_config

            # Display the current configuration
            if st.session_state["ml_config"]:
                st.write("Current ML Configuration:")
                st.json(st.session_state["ml_config"])

            # Load model state_dict() of trained models
            if st.session_state["ml_config"]:
                ml_config = st.session_state["ml_config"]
                ml_model = DeepGPModeler(
                    input_dims=ml_config["input_dims"],
                    hidden_dims=ml_config["hidden_dims"],
                    num_inducing_per_layer=ml_config["num_inducing_per_layer"],
                    final_outputs_dim=ml_config["final_outputs_dim"],
                    mean_type=ml_config["mean_type"],
                )

                model_state_dict_file = st.file_uploader("Upload a PyTorch model (.pth file)", type=["pth"])
                if model_state_dict_file is not None:
                    try:
                    # Read uploaded file into a BytesIO buffer
                        buffer = io.BytesIO(model_state_dict_file.read())
                        buffer.seek(0)

                        # Load state dict with device mapping
                        state_dict = torch.load(buffer, map_location=device)

                        # Move model to correct device before loading weights
                        ml_model.model.to(device)
                        ml_model.model.load_state_dict(state_dict)

                        st.success("Model state dictionary loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading .pth file: {e}")
                else:
                    st.warning("Please upload a .pth file.")

    ##############################################
    # Step 3: Run Predictions
    ##############################################
    with tab3:
        st.header("Step 3-1: Run Predictions")
        if uploaded_unknown_points is not None:
            # Select input features for unknown data points
            input_columns = st.multiselect(
                "Select input features:",
                options=unknown_points_df.columns,
                default=unknown_points_df.columns[0:-1],
            )
            output_columns = st.multiselect(
                "Select outputs:",
                options=unknown_points_df.columns,
                default=unknown_points_df.columns[-1],
            )
            if input_columns:
                if st.button("Confirm input columns"):
                    st.session_state.input_confirmed = True
                if st.session_state.input_confirmed:
                    unknown_x = unknown_points_df[input_columns].values
                    unknown_x = torch.tensor(unknown_x, dtype=torch.float32)
                    if output_columns:
                        unknown_y = unknown_points_df[output_columns].values
                        _, num_cols = unknown_y.shape
                        if num_cols == 1:
                            unknown_y = unknown_y.squeeze(-1)  # Remove additional dim if it is a column vector
                        unknown_y = torch.tensor(unknown_y, dtype=torch.float32)
                    else:
                        if ml_config.get("final_outputs_dim") is None:
                            ml_config["final_outputs_dim"] = 1
                        unknown_y = torch.zeros([len(unknown_x), ml_config["final_outputs_dim"]])
                        output_columns = [f"output_{x}" for x in range(ml_config["final_outputs_dim"])]
                        _, num_cols = unknown_y.shape
                        if num_cols == 1:
                            unknown_y = unknown_y.squeeze(-1)
                        unknown_y = torch.tensor(unknown_y, dtype=torch.float32)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    unknown_x = unknown_x.to(device)
                    unknown_y = unknown_y.to(device)

                    # Create TensorDataset and DataLoader
                    unknown_dataset = TensorDataset(unknown_x, unknown_y)
                    unknown_loader = DataLoader(unknown_dataset, batch_size=1024, shuffle=False)
                    st.write(f"Number of samples to predict: {len(unknown_loader.dataset)}")
                    if data_modelling_method == "deepGP":
                        # Prediction of unknown data points using trained models
                        if st.button("Run Predictions"):
                            st.session_state.run_predictions = True

                        if st.session_state.run_predictions:
                            with st.spinner("Running predictions..."):
                                ml_model.model.eval()
                                outputs_mean, outputs_var = ml_model.model.predict(unknown_x)

                                st.write("Predicting the unknown points!")
                                unknown_points_df["predicted_mean"] = outputs_mean.cpu().detach().numpy()
                                unknown_points_df["predicted_var"] = outputs_var.cpu().detach().numpy()

                                st.session_state["unknowninput_predictedmeanvar"] = unknown_points_df
                                st.dataframe(unknown_points_df)

                                if normalised_outputs and normaliser:
                                    output_normalisation_method = {}
                                    normalisation_options = ["none", "minmax", "standard", "robust", "l2", "log", "power"]
                                    for output_col in output_columns:
                                        # For each output column, create a selectbox to choose its normalisation method
                                        output_normalisation_method[output_col] = st.selectbox(
                                                                        f"Normalisation method for {output_col}:",
                                                                        options=normalisation_options,
                                                                        index=0,  # Default to the first option
                                                                        key=f"norm_{output_col}"  # Unique key for each selectbox
                                                                    )
                                        st.session_state.output_normalisation_method = output_normalisation_method

                                    if st.button("Submit normalisation method(s) for output(s)"):
                                        st.session_state.submit_output_normalisation = True
                                    if st.session_state.submit_output_normalisation:
                                        mean_var_original_df = normaliser.inverse_transform_gps_outputs(
                                            unknown_points_df["predicted_mean"],
                                            unknown_points_df["predicted_var"],
                                            output_columns,
                                            output_normalisation_method,
                                        )
                                        ong_outputs_df = normaliser.inverse_transform(unknown_points_df[output_columns], 
                                                                                    output_columns, 
                                                                                    output_normalisation_method)

                                        st.session_state["predicted_original_mean_var"] = mean_var_original_df
                                        unknown_points_df = pd.concat([unknown_points_df, ong_outputs_df, mean_var_original_df], axis=1)
                                        st.success("Predicted transformed output(s) inversed to original scales!")
                                        st.dataframe(unknown_points_df)
                                        st.session_state["raw_unknown_scaled_data"] = unknown_points_df
                                else:
                                    st.session_state["predicted_original_mean_var"] = unknown_points_df["predicted_mean", "predicted_var",]
                                    st.session_state["raw_unknown_data"] = pd.concat([st.session_state["raw_unknown_data"], unknown_points_df["predicted_mean", "predicted_var",]], axis=1)
                                    st.success("Predictions completed!")
                                    st.dataframe(unknown_points_df)
                                
            # Visualise Predictions
            if st.checkbox("Visualise Predictions"):

                # Allow users to choose plot types for predictions
                prediction_plot_types = st.multiselect(
                    "Select plot types for Predictions:",
                    options=["parity", "histogram", "cdf"],
                    default=["parity", "histogram", "cdf"]
                )
                
                # Select variables for prediction visualisation
                actual_varible = st.selectbox (
                    "Select the actual variable for visualisation:",
                    key = "actual_selectbox" ,
                    options = unknown_points_df.columns,
                )
                predicted_mean_varible = st.selectbox (
                    "Select the predicted mean variable for visualisation:",
                    key = "predicted_mean_selectbox" ,
                    options = unknown_points_df.columns,
                )
                predicted_var_varible = st.selectbox (
                    "Select the predicted variance of variable for visualisation:",
                    key = "predicted_var_selectbox" ,
                    options = unknown_points_df.columns,
                )

                if predicted_mean_varible is not None:
                    if normalised_outputs and normaliser:
                        actual = st.session_state["raw_unknown_scaled_data"][actual_varible].to_numpy().ravel()
                        predicted_mean = st.session_state["raw_unknown_scaled_data"][predicted_mean_varible].to_numpy().ravel()
                        predicted_var = st.session_state["raw_unknown_scaled_data"][predicted_var_varible].to_numpy().ravel()
                    else:
                        actual = st.session_state["raw_unknown_data"][actual_varible].to_numpy().ravel()
                        predicted_mean = st.session_state["raw_unknown_data"][predicted_mean_varible].to_numpy().ravel()
                        predicted_var = st.session_state["raw_unknown_data"][predicted_var_varible].to_numpy().ravel()
                    # Visualise predictions
                    visualise_predictions(
                        actual=actual,
                        predicted_mean=predicted_mean,
                        predicted_var=predicted_var,
                        plot_types=prediction_plot_types
                    )                    
        else:
            st.warning("Please upload a CSV file for input features of unknown datapoints.")

    ##############################################
    # Step 4: Forward UQ and Sensitivity Analysis
    ##############################################
    with tab4:
        st.header("Step 3-2: Forward UQ and Sensitivity Analysis")
        if uploaded_unknown_points is not None:
            # Forward UQ
            if st.checkbox("Perform Forward Uncertainty Quantification"):

                st.warning("Here the range/statistcs of loaded raw data will be used as the default prior of inputs in sampling! \
                    If the model is trained with normalised/scaled data, \
                    the code will inverse the transformation automatically!")

                num_samples = st.number_input("Number of Monte-Carlo samples", min_value=100, value=1000)
                unknown_points = st.session_state["raw_unknown_data"]
                samples_df = forward_uq(unknown_points, ml_model, input_columns, output_columns,  normaliser=normaliser, num_samples=num_samples)
                st.write("Forward UQ Results:")
                st.dataframe(samples_df)

                if st.button("Visualise forward UQ results"):
                    st.session_state.visualise_forward_uq = True

                if st.session_state.visualise_forward_uq:
                    # Allow users to choose plot types for forward UQ
                    forward_uq_plot_types = st.multiselect(
                        "Select plot types for Forward UQ:",
                        options=["histogram", "cdf", "boxplot", "violin", "errorbar", 
                                 "uncertainty_band", "scatter_density", "heatmap", "sobol", "spaghetti"],
                        default=["histogram", "cdf"]
                    )
                    # Visualise forward UQ results
                    visualise_forward_uq(samples_df, plot_types=forward_uq_plot_types)

            # Sensitivity Analysis
            if st.checkbox("Perform Sensitivity Analysis"):
                st.warning("Here the range of loaded raw data will be used for sampling in this sensitive analysis! \
                            If the model is trained with normalised/scaled data, \
                            the code will handle it automatically!")
                value = 1024
                num_samples = st.number_input("Number of Monte-Carlo samples for sensitivity analysis (prefer 2^n)", min_value=100, value=value)

                unknown_points = st.session_state["raw_unknown_data"]

                morris_results, sobol_results = sensitivity_analysis(unknown_points, ml_model, input_columns, output_columns, normaliser=normaliser, N=num_samples)

                # Allow users to choose plot types for sensitivity analysis
                sensitivity_plot_types = st.multiselect(
                    "Select plot types for Sensitivity Analysis:",
                    options=["morris", "sobol"],
                    default=["morris", "sobol"]
                )

                # Visualise sensitivity analysis results
                visualisation_checkbox = False
                if morris_results and sobol_results:  # Checks if both dictionaries are non-empty
                    visualise_sensitivity_analysis(morris_results, sobol_results, input_columns)

# Run the app
if __name__ == "__main__":
    app()