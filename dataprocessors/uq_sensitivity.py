import numpy as np
import scipy.stats as stats
import pandas as pd
import torch
from SALib.sample import saltelli, morris
from SALib.analyze import sobol, morris as morris_analyze
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def forward_uq(unknown_points_df, ml_model, input_columns=None, output_columns=None, normaliser=None, num_samples=2^10):
    """
    Perform forward uncertainty quantification using Monte-Carlo sampling.
    """
    # Initialise sample_df
    samples_df = None
    raw_samples_df = None
    if "input_priors_confirmed" not in st.session_state:
        st.session_state.input_priors_confirmed = False

    # Dictionary of default PDFs
    default_pdfs = {
        "Normal": stats.norm,
        "Uniform": stats.uniform,
        "Log-Normal": stats.lognorm,
    }

    # Check if input_columns exist or not
    if input_columns is None:
        input_columns = st.multiselect(
                    "Select input features:",
                    options=unknown_points_df.columns,
                    default=unknown_points_df.columns[0:-1],
                )
    # Check if output_columns exist or not
    if output_columns is None:
        output_columns = st.multiselect(
                    "Select outputs:",
                    options=unknown_points_df.columns,
                    default=unknown_points_df.columns[-1],
                )

    # Sample from the prior distributions
    samples = {}
    for col in input_columns:
        st.write(f"Select a prior distribution for {col}:")
        pdf_type = st.selectbox(f"PDF type for {col}", list(default_pdfs.keys()))
        
        if pdf_type == "Normal":
            mean = st.number_input(f"Mean for {col}", value=unknown_points_df[col].mean())
            std = st.number_input(f"Standard Deviation for {col}", value=unknown_points_df[col].std())
            samples[col] = default_pdfs[pdf_type](loc=mean, scale=std).rvs(num_samples)
        elif pdf_type == "Uniform":
            low = st.number_input(f"Lower bound for {col}", value=unknown_points_df[col].min())
            high = st.number_input(f"Upper bound for {col}", value=unknown_points_df[col].max())
            samples[col] = default_pdfs[pdf_type](loc=low, scale=high - low).rvs(num_samples)
        elif pdf_type == "Log-Normal":
            mean = st.number_input(f"Mean for {col}", value=unknown_points_df[col].mean())
            std = st.number_input(f"Standard Deviation for {col}", value=unknown_points_df[col].std())
            samples[col] = default_pdfs[pdf_type](s=std, scale=np.exp(mean)).rvs(num_samples)

    if st.button("Confirm input priors"):
        st.session_state.input_priors_confirmed = True
    
    if st.session_state.input_priors_confirmed:
        # Only for the normalisation purpose!
        for col in output_columns:
            samples[col] = np.zeros(num_samples)

        raw_samples_df = pd.DataFrame(samples)

        # Check if normalisation is needed
        normalised_data = st.checkbox("Normalisation similar to the training is needed?", key="normalised_data")

        # if the model is built upon the normalised data the samples needed to be normalised as well.
        # Option1: checkbox of the normalised data is ticked and normaliser is missing!
        if normalised_data and normaliser is None:
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

                samples_df = pd.DataFrame(samples)
                samples_df = normaliser.apply_scalers(samples_df)
        # Option2: the normaliser has already been loaded in the previous sessions
        elif normalised_data and normaliser is not None:
            samples_df = pd.DataFrame(samples)
            samples_df = normaliser.apply_scalers(samples_df)
        else:
            samples_df = pd.DataFrame(samples)

        # Propagate uncertainty through the model
        st.write("Propagating uncertainty through the model...")
        outputs_mean, outputs_var = [], []
        for _, row in samples_df[input_columns].iterrows():
            x = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
            # Option 1: If model has a .device property
            if hasattr(ml_model.model, 'device'):
                x = x.to(ml_model.model.device)
            # Option 2: Fallback (for PyTorch models)
            else:
                model_device = next(ml_model.model.parameters()).device
                x = x.to(model_device)

            mean, var = ml_model.model.predict(x)
            outputs_mean.append(mean.item())
            outputs_var.append(var.item())

        # Store results
        # Current version only work for uni-ouput problems
        samples_df["predicted_mean"] = outputs_mean
        samples_df["predicted_var"] = outputs_var

        raw_samples_df["predicted_mean"] = outputs_mean
        raw_samples_df["predicted_var"] = outputs_var

        if normaliser:
            output_normalisation_method = {}
            normalisation_options = ["none", "minmax", "standard", "robust", "l2", "log", "power"]
            for output_col in output_columns:
                # For each output column, create a selectbox to choose its normalisation method
                output_normalisation_method[output_col] = st.selectbox(
                                                            f"Normalisation method for {output_col} (for recovering the original scales):",
                                                            options=normalisation_options,
                                                            index=0,  # Default to the first option
                                                            key=f"norm_{output_col}_1"  # Unique key for each selectbox
                                                        )

                if output_normalisation_method[output_col] != "none":
                    mean_var_original_df = normaliser.inverse_transform_gps_outputs(
                                                        samples_df["predicted_mean"],
                                                        samples_df["predicted_var"],
                                                        output_columns,
                                                        output_normalisation_method,
                                                    )
                    # Current version only work for uni-ouput problems
                    raw_samples_df["predicted_mean"] = mean_var_original_df.iloc[:, 0].values  # 1st column
                    raw_samples_df["predicted_var"] = mean_var_original_df.iloc[:, 1].values  # 2nd column
        
        # Delet the extra added columns in the sample_df for normaliation purpose only!
        for col in output_columns:
            if col in raw_samples_df.columns:
                del raw_samples_df[col]

    return raw_samples_df

def visualise_forward_uq(samples_df, plot_types=None):
    """
    Visualize forward uncertainty quantification results using various plot types.
    
    Args:
        samples_df (pd.DataFrame): DataFrame containing Monte-Carlo samples and predictions.
        plot_types (list): List of plot types to generate. Defaults to all available plots.
                           Options: ["histogram", "cdf", "boxplot", "violin", "errorbar", 
                                    "uncertainty_band", "scatter_density", "heatmap", "sobol", "spaghetti"]
    """
    if plot_types is None:
        plot_types = ["histogram", "cdf", "boxplot", "violin", "errorbar", 
                      "uncertainty_band", "scatter_density", "heatmap", "sobol", "spaghetti"]

    st.write("### Forward Uncertainty Quantification Results")

    # 1. Histograms & Probability Density Functions (PDFs)
    if "histogram" in plot_types:
        st.write("#### Histograms & PDFs")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(samples_df["predicted_mean"], kde=True, ax=ax[0])
        ax[0].set_title("Distribution of Predicted Means")
        ax[0].set_xlabel("Predicted Mean")
        ax[0].set_ylabel("Frequency")

        sns.histplot(samples_df["predicted_var"], kde=True, ax=ax[1])
        ax[1].set_title("Distribution of Predicted Variances")
        ax[1].set_xlabel("Predicted Variance")
        ax[1].set_ylabel("Frequency")
        st.pyplot(fig)

    # 2. Cumulative Distribution Functions (CDFs)
    if "cdf" in plot_types:
        st.write("#### Cumulative Distribution Functions (CDFs)")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.ecdfplot(samples_df["predicted_mean"], ax=ax[0])
        ax[0].set_title("CDF of Predicted Means")
        ax[0].set_xlabel("Predicted Mean")
        ax[0].set_ylabel("Cumulative Probability")

        sns.ecdfplot(samples_df["predicted_var"], ax=ax[1])
        ax[1].set_title("CDF of Predicted Variances")
        ax[1].set_xlabel("Predicted Variance")
        ax[1].set_ylabel("Cumulative Probability")
        st.pyplot(fig)

    # 3. Box Plots / Violin Plots
    if "boxplot" in plot_types:
        st.write("#### Box Plots")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=samples_df[["predicted_mean", "predicted_var"]], ax=ax)
        ax.set_title("Box Plots of Predicted Means and Variances")
        ax.set_ylabel("Value")
        st.pyplot(fig)

    if "violin" in plot_types:
        st.write("#### Violin Plots")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=samples_df[["predicted_mean", "predicted_var"]], ax=ax)
        ax.set_title("Violin Plots of Predicted Means and Variances")
        ax.set_ylabel("Value")
        st.pyplot(fig)

    # 4. Error Bars / Confidence Intervals
    if "errorbar" in plot_types:
        st.write("#### Error Bars / Confidence Intervals")
        mean = samples_df["predicted_mean"].mean()
        std = samples_df["predicted_mean"].std()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(x=0, y=mean, yerr=2 * std, fmt="o", capsize=5)
        ax.set_title("Mean ± 2σ Confidence Interval")
        ax.set_xticks([])
        ax.set_ylabel("Predicted Mean")
        st.pyplot(fig)

    # 5. Uncertainty Bands / Envelopes
    if "uncertainty_band" in plot_types:
        st.write("#### Uncertainty Bands / Envelopes")
        time = np.arange(len(samples_df))
        mean = samples_df["predicted_mean"].rolling(window=10).mean()
        std = samples_df["predicted_mean"].rolling(window=10).std()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, mean, label="Mean")
        ax.fill_between(time, mean - 2 * std, mean + 2 * std, alpha=0.3, label="±2σ")
        ax.set_title("Uncertainty Band Around Mean")
        ax.set_xlabel("Time")
        ax.set_ylabel("Predicted Mean")
        ax.legend()
        st.pyplot(fig)

    # 6. Scatter Plots with Density Shading
    if "scatter_density" in plot_types:
        st.write("#### Scatter Plots with Density Shading")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(x=samples_df["predicted_mean"], y=samples_df["predicted_var"], fill=True, cmap="Blues", ax=ax)
        ax.set_title("Scatter Plot with Density Shading")
        ax.set_xlabel("Predicted Mean")
        ax.set_ylabel("Predicted Variance")
        st.pyplot(fig)

    # 7. Heatmaps / Contour Plots
    if "heatmap" in plot_types:
        st.write("#### Heatmaps / Contour Plots")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(x=samples_df["predicted_mean"], y=samples_df["predicted_var"], bins=50, cmap="viridis", ax=ax)
        ax.set_title("Heatmap of Predicted Means vs. Variances")
        ax.set_xlabel("Predicted Mean")
        ax.set_ylabel("Predicted Variance")
        st.pyplot(fig)

    # 8. Spaghetti Plots
    if "spaghetti" in plot_types:
        st.write("#### Spaghetti Plots")
        time = np.arange(len(samples_df))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(10):  # Plot first 10 samples
            ax.plot(time, samples_df["predicted_mean"].iloc[i:i+100], alpha=0.5)
        ax.set_title("Spaghetti Plot of Predicted Means")
        ax.set_xlabel("Time")
        ax.set_ylabel("Predicted Mean")
        st.pyplot(fig)

def sensitivity_analysis(unknown_points_df, ml_model, input_columns=None, output_columns=None, normaliser=None, N=2^10):
    """
    Perform sensitivity analysis using Morris, Sobol, and local sensitivity methods.
    """
    samples_df = None
    raw_samples_df = None
    morris_Si = {} 
    sobol_Si = {}

    # Initialize session state for bounds if it doesn't exist
    if 'param_bounds' not in st.session_state:
        st.session_state.param_bounds = {}

    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    if "normalised_data_ckeckbox" not in st.session_state:
        st.session_state.normalised_data_ckeckbox = False

    if 'sen_submit_button' not in st.session_state:
        st.session_state.sen_submit_button = False

    st.write("### Sensitivity Analysis")
    # Create a form for all sliders
    with st.form("parameter_bounds"):
        st.write("### Define Parameter Ranges")
    
        bounds = []
        for col in input_columns:
            col_min = unknown_points_df[col].min()
            col_max = unknown_points_df[col].max()
        
            # Use the previously selected bounds if they exist
            default_min = st.session_state.param_bounds.get(col, [col_min, col_max])[0]
            default_max = st.session_state.param_bounds.get(col, [col_min, col_max])[1]
        
            # Create a range slider for each parameter
            min_val, max_val = st.select_slider(
                f"Range for {col}",
                options=sorted(unknown_points_df[col].unique()),
                value=(default_min, default_max)
            )
            bounds.append([min_val, max_val])
            st.session_state.param_bounds[col] = [min_val, max_val]
    
        submitted = st.form_submit_button("Update Bounds")

    # Define the problem with user-selected bounds
    problem = {
        "num_vars": len(input_columns),
        "names": input_columns,
        "bounds": [st.session_state.param_bounds[col] for col in input_columns] if submitted else bounds
    }

    # Now you can use this problem definition for sampling
    if submitted:
        st.session_state.form_submitted = True
    if st.session_state.form_submitted:
        st.write("Current sensitivity problem:")
        st.json(problem)

        ###############################
        # Morris Sensitivity analysis #
        ###############################
        st.write("#### Morris Sensitivity Analysis")
        model_device = next(ml_model.model.parameters()).device

        # Generate parameter samples
        param_values = morris.sample(problem, N=N)
        # Assign param_values to sample dictionary with relevant columns:
        samples = {}
        for i, col in enumerate(input_columns):
            samples[col] = param_values[:, i]  # All rows for column i

        for col in output_columns:
            samples[col] = np.zeros(len(param_values))

        raw_samples_df = pd.DataFrame(samples)

        # Scale/normalise the param values if required
        # Check if normalisation is needed
        normalised_data = st.checkbox("Normalisation similar to the training is needed?", key="normalised_data_1")
        if normalised_data:
            st.session_state.normalised_data_ckeckbox = True

        if normaliser is not None:
            st.session_state.normalised_available = True
        else:
            st.session_state.normalised_available = False

        # if the model is built upon the normalised data the samples needed to be normalised as well.
        # Option1: checkbox of the normalised data is ticked and normaliser is missing!
        if st.session_state.normalised_data_ckeckbox and normaliser is None:
            saved_scalers_file = st.file_uploader("Upload file of the saved scaler", type=["pkl"])
            st.sidebar.warning("If you applied normalisation to data before, " \
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

                samples_df = pd.DataFrame(samples)
                samples_df = normaliser.apply_scalers(samples_df)
                output_normalisation_method = {}
                normalisation_options = ["none", "minmax", "standard", "robust", "l2", "log", "power"]
                for output_col in output_columns:
                # For each output column, create a selectbox to choose its normalisation method
                    output_normalisation_method[output_col] = st.selectbox(
                                                                f"Normalisation method for {output_col} (for recovering the original scales):",
                                                                options=normalisation_options,
                                                                index=0,  # Default to the first option
                                                                key=f"norm_{output_col}_2"  # Unique key for each selectbox
                                                            )
        # Option2: the normaliser has already been loaded in the previous sessions
        elif st.session_state.normalised_data_ckeckbox and normaliser is not None:
            samples_df = pd.DataFrame(samples)
            samples_df = normaliser.apply_scalers(samples_df)
            output_normalisation_method = {}
            normalisation_options = ["none", "minmax", "standard", "robust", "l2", "log", "power"]
            for output_col in output_columns:
                # For each output column, create a selectbox to choose its normalisation method
                output_normalisation_method[output_col] = st.selectbox(
                                                            f"Normalisation method for {output_col} (for recovering the original scales):",
                                                            options=normalisation_options,
                                                            index=0,  # Default to the first option
                                                            key=f"norm_{output_col}_3"  # Unique key for each selectbox
                                                        )
        else:
            samples_df = pd.DataFrame(samples)

        if st.button("Start sensitivity analysis"):
            st.session_state.sen_submit_button = True

        if st.session_state.sen_submit_button:
            # Propagate samples through the model
            #st.write("Propagating uncertainty through the model...")
            outputs_mean, outputs_var = [], []
            for _, row in samples_df[input_columns].iterrows():
                x = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
                # Option 1: If model has a .device property
                if hasattr(ml_model.model, 'device'):
                    x = x.to(ml_model.model.device)
                # Option 2: Fallback (for PyTorch models)
                else:
                    model_device = next(ml_model.model.parameters()).device
                    x = x.to(model_device)

                mean, var = ml_model.model.predict(x)
                outputs_mean.append(mean.item())
                outputs_var.append(var.item())

            # Store results
            # Current version only work for uni-ouput problems
            samples_df["predicted_mean"] = outputs_mean
            samples_df["predicted_var"] = outputs_var

            raw_samples_df["predicted_mean"] = outputs_mean
            raw_samples_df["predicted_var"] = outputs_var

            # Initialize an empty array for outputs
            Y = np.zeros(len(param_values))

            # Option 1: inverse the transform (scale/normalise) of the model predict means if normaliser is available
            if normalised_data:
                for output_col in output_columns:
                    if output_normalisation_method[output_col] != "none":
                        mean_var_original_df = normaliser.inverse_transform_gps_outputs(
                                                            samples_df["predicted_mean"],
                                                            samples_df["predicted_var"],
                                                            output_columns,
                                                            output_normalisation_method,
                                                        )
                        # Current version only work for uni-ouput problems
                        Y = np.array(mean_var_original_df.iloc[:, 0].values)  # 1st column
            # Option 2: normaliser is not None then the model prediction is the right scale of the raw data 
            else:
                Y = np.array(raw_samples_df["predicted_mean"].values)

            morris_Si = morris_analyze.analyze(problem, param_values, Y)
            st.write("Morris Sensitivity Indices:")
            st.json(morris_Si)

            #################
            # Sobol Indices #
            #################
            st.write("#### Sobol Sensitivity Analysis")
            param_values = saltelli.sample(problem, N)

            # Assign param_values to sample dictionary with relevant columns:
            samples = {}
            for i, col in enumerate(input_columns):
                samples[col] = param_values[:, i]  # All rows for column i

            for col in output_columns:
                samples[col] = np.zeros(len(param_values))

            raw_samples_df = pd.DataFrame(samples)

            # Scale/normalise the param values if required
            # if the model is built upon the normalised data the samples needed to be normalised as well.
            # Option1: checkbox of the normalised data is ticked and normaliser is missing!
            if normalised_data and normaliser is None:
                saved_scalers_file = st.file_uploader("Upload file of the saved scaler", type=["pkl"])
                st.sidebar.warning("If you applied normalisation to data before, " \
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

                    samples_df = pd.DataFrame(samples)
                    samples_df = normaliser.apply_scalers(samples_df)
            # Option2: the normaliser has already been loaded in the previous sessions
            elif normalised_data and normaliser is not None:
                samples_df = pd.DataFrame(samples)
                samples_df = normaliser.apply_scalers(samples_df)
            else:
                samples_df = pd.DataFrame(samples)

            # Propagate samples through the model
            #st.write("Propagating uncertainty through the model...")
            outputs_mean, outputs_var = [], []
            for _, row in samples_df[input_columns].iterrows():
                x = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
                # Option 1: If model has a .device property
                if hasattr(ml_model.model, 'device'):
                    x = x.to(ml_model.model.device)
                # Option 2: Fallback (for PyTorch models)
                else:
                    model_device = next(ml_model.model.parameters()).device
                    x = x.to(model_device)

                mean, var = ml_model.model.predict(x)
                outputs_mean.append(mean.item())
                outputs_var.append(var.item())

            # Store results
            # Current version only work for uni-ouput problems
            samples_df["predicted_mean"] = outputs_mean
            samples_df["predicted_var"] = outputs_var

            raw_samples_df["predicted_mean"] = outputs_mean
            raw_samples_df["predicted_var"] = outputs_var

            # Initialize an empty array for outputs
            Y = np.zeros(len(param_values))

            # Option 1: inverse the transform (scale/normalise) of the model predict means if normaliser is available
            if normalised_data:
                for output_col in output_columns:
                    if output_normalisation_method[output_col] != "none":
                        mean_var_original_df = normaliser.inverse_transform_gps_outputs(
                                                            samples_df["predicted_mean"],
                                                            samples_df["predicted_var"],
                                                            output_columns,
                                                            output_normalisation_method,
                                                        )
                        # Current version only work for uni-ouput problems
                        Y = np.array(mean_var_original_df.iloc[:, 0].values)  # 1st column
            # Option 2: normaliser is not None then the model prediction is the right scale of the raw data 
            else:
                Y = np.array(raw_samples_df["predicted_mean"].values)

            sobol_Si = sobol.analyze(problem, Y)
            st.write("Sobol Indices:")
            st.json(sobol_Si)

    '''
    # Local Sensitivity (Inverse Gradient): Subjected to be developed
    st.write("#### Local Sensitivity Analysis")
    
    '''
    return morris_Si, sobol_Si

def visualise_sensitivity_analysis(morris_results, sobol_results, input_columns, gradients_df=None):
    """
    Visualize sensitivity analysis results using key plots for Morris, Sobol, and local sensitivity methods.
    
    Args:
        morris_results (dict): Results from Morris sensitivity analysis.
        sobol_results (dict): Results from Sobol sensitivity analysis.
        gradients_df (pd.DataFrame): DataFrame containing local sensitivity gradients.
    """
    st.write("### Sensitivity Analysis Results")

    # 1. Morris: μ* vs σ scatter plot
    st.write("#### Morris Sensitivity: μ* vs σ Scatter Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(morris_results["mu_star"], morris_results["sigma"])
    for i, name in enumerate(morris_results["names"]):
        ax.text(morris_results["mu_star"][i], morris_results["sigma"][i], name, fontsize=9)
    ax.set_xlabel("μ* (Importance)")
    ax.set_ylabel("σ (Non-linearity/Interactions)")
    ax.set_title("Morris Sensitivity: μ* vs σ")
    st.pyplot(fig)

    # 2. Sobol: Bar plots of S1, ST (and S2)
    st.write("#### Sobol Sensitivity:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sobol_df = pd.DataFrame({
        "Feature": input_columns,
        "First Order (S1)": sobol_results["S1"],
        "Total Order (ST)": sobol_results["ST"],
    })
    sobol_df.set_index("Feature").plot(kind="bar", ax=ax)
    ax.set_title("Sobol Sensitivity Indices")
    ax.set_ylabel("Sensitivity Index")
    st.pyplot(fig)

    S2 = sobol_results["S2"]
    # Create masked array for visualization
    mask = np.isnan(S2)
    S2_masked = np.ma.masked_where(mask, S2)
    param_names = input_columns

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Create heatmap
    heatmap = sns.heatmap(S2_masked, 
                         annot=True, 
                         fmt=".3f",
                         cmap="coolwarm", 
                         center=0,
                         xticklabels=param_names,
                         yticklabels=param_names,
                         mask=mask,
                         cbar_kws={'label': 'Second-Order Index (S2)'},
                         ax=ax)  # Important for Streamlit
    # Add title and formatting
    ax.set_title("Second-Order Sobol Indices (Interaction Effects)")
    # Display in Streamlit
    st.pyplot(fig)

    '''
    # 3. Local Sensitivity: Tornado / Spider plots
    st.write("#### Local Sensitivity: Tornado Plot")
    gradients_df = gradients_df.abs().mean().sort_values(ascending=False)  # Use absolute mean gradients
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=gradients_df.values, y=gradients_df.index, ax=ax, orient="h")
    ax.set_title("Local Sensitivity: Tornado Plot")
    ax.set_xlabel("Absolute Mean Gradient")
    ax.set_ylabel("Input Feature")
    st.pyplot(fig)
    '''
def visualise_predictions(actual, predicted_mean, predicted_var=None, plot_types=None):
    """
    Visualize predictions against the original data using selected plot types.
    
    Args:
        actual (np.array or pd.Series): Actual values (ground truth).
        predicted_mean (np.array or pd.Series): Predicted mean values.
        predicted_var (np.array or pd.Series, optional): Predicted variances. Required for confidence intervals.
        plot_types (list): List of plot types to generate. Defaults to all available plots.
                           Options: ["parity", "histogram", "cdf"]
    """
    if plot_types is None:
        plot_types = ["parity", "histogram", "cdf"]

    st.write("### Model Predictions Visualization")

    # 1. Parity Plot (Prediction vs. Actual)
    if "parity" in plot_types:
        if predicted_var is not None:
            st.write("#### Parity Plot (Prediction vs. Actual)")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = sns.scatterplot(x=actual, y=predicted_mean, ax=ax, alpha=0.6)
            predicted_std = np.sqrt(predicted_var)
            # Get the current color from the scatterplot
            color = scatter.collections[0].get_facecolor()[0]  # Get the first color used

            # Add error bars (assuming `predicted_std` contains the standard deviation)
            ax.errorbar(
                x=actual,
                y=predicted_mean,
                yerr= 1.96* predicted_std,  # Error bar length (can be a single value, array, or 2xN array for asymmetric errors)
                fmt='none',          # Do not plot markers (since scatterplot already handles it)
                ecolor=color,        # Error bar color
                alpha=0.4,           # Transparency
                capsize=3,           # Size of error bar caps
                label="Error (±2 SD)"
            )
            ax.plot([min(actual), max(actual)], [min(actual), max(actual)], color="red", linestyle="--", label="Ideal")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Parity Plot: Prediction vs. Actual")
            ax.legend()
            st.pyplot(fig)

        else:
            st.write("#### Parity Plot (Prediction vs. Actual)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=actual, y=predicted_mean, ax=ax, alpha=0.6)
            ax.plot([min(actual), max(actual)], [min(actual), max(actual)], color="red", linestyle="--", label="Ideal")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Parity Plot: Prediction vs. Actual")
            ax.legend()
            st.pyplot(fig)

    # 2. Histogram of Error Ratios (including confidence intervals)
    if "histogram" in plot_types:
        if predicted_var is not None:
            st.write("#### Histogram of Error Ratios with Confidence Intervals")
            error_ratios = (predicted_mean - actual) / actual
            confidence_intervals = 1.96 * np.sqrt(predicted_var) / actual  # 95% confidence interval

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(error_ratios, kde=True, ax=ax, label="Error Ratios")
            #ax.axvline(error_ratios.mean(), color="red", linestyle="--", label="Mean Error Ratio")
            ax.fill_betweenx(
                y=ax.get_ylim(),
                x1=error_ratios.mean() - confidence_intervals.mean(),
                x2=error_ratios.mean() + confidence_intervals.mean(),
                color="gray",
                alpha=0.3,
                label="95% Confidence Interval",
            )
            ax.set_xlabel("Error Ratio")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Error Ratios")
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("#### Histogram of Error Ratios with Confidence Intervals")
            error_ratios = (predicted_mean - actual) / actual
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(error_ratios, kde=True, ax=ax, label="Error Ratios")
            ax.set_xlabel("Error Ratio")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Error Ratios")
            ax.legend()
            st.pyplot(fig)

    # 3. Empirical Cumulative Distribution Function (CDF) of Mean Values
    if "cdf" in plot_types:
        st.write("#### Empirical CDF of Mean Values")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.ecdfplot(actual, ax=ax, label="Actual")
        sns.ecdfplot(predicted_mean, ax=ax, label="Predicted")
        ax.set_xlabel("Values")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Empirical CDF: Actual vs. Predicted")
        ax.legend()
        st.pyplot(fig)