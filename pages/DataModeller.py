import streamlit as st
import pandas as pd
import torch
import os
import json
from bayesmodels import (
    Customised_Loss,
    DeepGPModeler,
    DataHandler,
    TensorDataset,
    DataLoader,
)

def app():
    st.title("SimuBayes: Simulation noisy data with Bayesian MLs")
    # Initialize session state for stop button
    if "stop_training" not in st.session_state:
        st.session_state.stop_training = False
    if "start_training" not in st.session_state:
        st.session_state.start_training = False
    if "enable_test_losses_check" not in st.session_state:
        st.session_state.enable_test_losses_check = False

    # retrive the directory for storing the outputs
    output_dir = st.session_state["outputs_directory"]

    # Dropdown to select dataset
    st.sidebar.markdown("---")  # Shortcut for <hr> in Markdown 
    st.sidebar.subheader("Data Handler")
    dataset_options = ["--- select an dataset ---", "Preprocessed Data", "Normalised Data", "Reduced Data"]
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)

    # Display appropriate plot based on selected dataset and plot type
    if selected_dataset == "Preprocessed Data":
        data_df = st.session_state.get("preprocessed_data", None)
        st.write("### Data to be modeled: Preprocessed Data")
        st.write(data_df)
    elif selected_dataset == "Normalised Data":
        data_df = st.session_state.get("normalised_data", None)
        st.write("### Data to be modeled: Normalised Data")
        st.write(data_df)
    elif selected_dataset == "Reduced Data":
        data_df = st.session_state.get("reduced_data", None)
        st.write("### Data to be modeled: Reduced Data")
        st.write(data_df)
    else:
        st.warning("Please select a dataset to proceed!!!")
    
    if selected_dataset == "Preprocessed Data" or selected_dataset == "Normalised Data" or selected_dataset == "Reduced Data" :
        input_columns = st.sidebar.multiselect(
                            "Select input features:",
                            options=data_df.columns,  # All columns are available for selection
                            default=None  # No columns selected by default
                            )
        outputs_columns = st.sidebar.multiselect(
                            "Select outputs:",
                            options=data_df.columns,  # All columns are available for selection
                            default=None  # No columns selected by default
                            )
                        
        if input_columns and outputs_columns:
            # Ask the user to input the split ratio as a comma-separated string
            split_ratio_input = st.sidebar.text_input("Split ratio (comma-separated)", "0.7,0.15,0.15")

            # Convert the input string to a list of floats
            try:
                split_ratio = [float(x) for x in split_ratio_input.split(",")]
                if len(split_ratio) != 3:
                    st.error("Please enter exactly 3 values for the split ratio (e.g., 0.7,0.15,0.15).")
                elif abs(sum(split_ratio) - 1.0) > 1e-6:
                    st.error("The split ratio must sum to 1.0.")
                else:
                    st.info(f"Split ratio set to: {split_ratio}")
            except ValueError:
                st.error("Invalid input. Please enter comma-separated numbers (e.g., 0.7,0.15,0.15).")

            # Initialize DataHandler
            data_handler = DataHandler(
                    data_df=data_df,
                    input_columns=input_columns,
                    outputs_columns=outputs_columns,
                    split_ratio=split_ratio
            )

            # Load and preprocess data
            X_train, y_train, X_val, y_val, X_test, y_test = data_handler.load_data()
            train_x, train_y, val_x, val_y, test_x, test_y = data_handler.preprocess_data(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )

            # Create DataLoader objects
            train_loader, val_loader, test_loader = data_handler.create_dataloaders(
                    train_x, train_y, val_x, val_y, test_x, test_y, batch_size=1024
                )

            # Display data shapes
            #st.write("### Data Shapes")
            #st.info(f"Train X: {train_x.shape}, Train Y: {train_y.shape}")
            #st.info(f"Test X: {test_x.shape}, Test Y: {test_y.shape}")
            #st.info(f"Validation X: {val_x.shape}, Validation Y: {val_y.shape}")
        else:
            st.warning("Please select input and outputs columns.")
    
    st.sidebar.markdown("---")  # Shortcut for <hr> in Markdown 
    st.sidebar.header("Model configuration")
    options=["--- select an approach ---", "deepGP"]
    data_modelling_method = st.sidebar.selectbox("Modelling Approach", options)
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None

    if data_modelling_method == "GP":
        st.warning("GP approach is under development!")
    elif data_modelling_method == "deepGP":
        input_dims = st.sidebar.number_input("Input Dimension", min_value=1, value=len(input_columns))
        hidden_dims = st.sidebar.text_input("Hidden Dimensions List (comma-separated)", "2, 2")
        num_inducing_per_layer = st.sidebar.text_input("Number of inducing points per hidden layer (comma-separated)", "64, 64, 64")
        final_outputs_dim = st.sidebar.number_input("Final outputs Dimension", min_value=1, value=len(outputs_columns))
        mean_type = st.sidebar.selectbox("Mean Type", ["zero", "constant"])
        
        if final_outputs_dim ==1:
            final_outputs_dim = None
            
        # Convert hidden_dims to list of integers
        hidden_dims = [int(x) for x in hidden_dims.split(",")]
        num_inducing_per_layer = [int(x) for x in num_inducing_per_layer.split(",")]

        if  len(num_inducing_per_layer) != (len(hidden_dims)+1):
            st.error("length of hidden dims list should be length of inducing point per layer plus 1!")
            st.stop()

        # Training Configuration
        st.sidebar.header("Training Configuration")
        num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, value=500)
        learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, value=0.001, step=0.0001, format="%0.5f")
        model_name = st.sidebar.text_input("Name of trained model (*.pth)", "trained_model.pth")

        enable_editing = st.sidebar.checkbox("Enable customised loss")
        if enable_editing:
            st.warning ("To enable the customised loss, you need to edit the \'Customised_Loss\' and \'train\' in \'DeepGPModeller\' in \'parametric_deepgp.py\'")
            custom_loss = Customised_Loss(multipliers, denominators, v, m, pred_weight=0.1) # this is inline with the example code!
        else:
            custom_loss = None

        deepgp_config = {
                        "data_modelling_method": data_modelling_method, # data modelling approach
                        "input_dims": input_dims,  # Fixed input dimensions
                        "hidden_dims": hidden_dims,  # Placeholder for hidden dimensions (will be replaced by hyperparams)
                        "num_inducing_per_layer": num_inducing_per_layer,  # number of inducing points per layer
                        "final_outputs_dim": final_outputs_dim, # dimension of final output layer
                        "mean_type": mean_type,  # Fixed mean type
                        "custom_loss": custom_loss, # custom loss functions
                        "num_epochs": num_epochs,  # Fixed number of epochs
                        "learning_rate": learning_rate,  # Fixed learning rate
                        }

        outputs_dir = st.session_state["outputs_directory"]
        st.warning(f"outputs_dir: {outputs_dir}")

        # Save the dictionary of deepgp model configuration to a '.pkl' file
        with open(os.path.join(outputs_dir, "ml_config.json"), "w") as file:
            json.dump(deepgp_config, file)

        st.session_state["ml_config"] = deepgp_config

        # Initialize DeepGPModel

        deepgp_model = DeepGPModeler(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_inducing_per_layer=num_inducing_per_layer,
            final_outputs_dim=final_outputs_dim,
            mean_type=mean_type,
            custom_loss=custom_loss  # Pass the custom loss function
        )
        # Train the model
        st.write("### Training Progress")

        if "loss_values" not in st.session_state:
            st.session_state.loss_values = []  # Persist losses across reruns
        if "loss_values_df" not in st.session_state:
            st.session_state.loss_values_df = pd.DataFrame()  # Persist losses_df across reruns
        if "test_rmse_values" not in st.session_state:
            st.session_state.test_rmse_values = []  # Persist RMSE across reruns
        if "test_msll_values" not in st.session_state:
            st.session_state.test_msll_values = []  # Persist MSLL across reruns
        if "test_metrics_df" not in st.session_state:
            st.session_state.test_metrics_df = pd.DataFrame()  # Persist test_metric_df across reruns
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Training"):
                st.session_state.start_training = True
                st.session_state.stop_training = False
                deepgp_model.reset_state()
                st.info("The trained model will be saved automatically upon completion of training.")
            test_loss_checkbox=st.checkbox("Report losses of test data")
            if test_loss_checkbox:
                st.session_state.enable_test_losses_check = True
        with col2:
            if st.button("Stop Training"):
                st.session_state.stop_training = True

        if st.session_state.get("start_training", False):
            deepgp_model.train(train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate, eval_every=10)
            # Save trained model at the end of training.
            st.session_state["trained_model"] = deepgp_model
            if st.session_state["trained_model"] is None:
                st.error("No model to save. Train it first!")
            else:
                trained_model_path = os.path.join(output_dir, model_name)
                torch.save(st.session_state["trained_model"].model.cpu().state_dict(), trained_model_path)
                st.success(f"Trained model saved as : {trained_model_path}")
        '''
        if st.sidebar.button("Save Trained Model", key="save_model"):
            if st.session_state["trained_model"] is None:
                st.error("No model to save. Train it first!")
            else:
                trained_model_path = os.path.join(output_dir, model_name)
                torch.save(st.session_state["trained_model"].model.cpu().state_dict(), trained_model_path)
                st.success(f"Trained model saved as : {trained_model_path}")
        '''
        if st.session_state.stop_training:
            st.session_state.start_training = False

    else:
        st.warning("No valid modelling approach is selected.")
        st.stop()
