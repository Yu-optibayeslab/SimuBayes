import streamlit as st
import matplotlib.pyplot as plt
from dataprocessors import Visualiser

def app():
    st.title("Visualisations")
    st.sidebar.header("Visualisation Options")

    # Retrieve data from session state
    data = st.session_state.get("processed_data", None)
    reduced_data = st.session_state.get("reduced_data", None)

    if data is None:
        st.warning("No processed data found. Please process data on the main page first.")
        st.stop()

    # Initialize Visualiser instance
    visualiser = Visualiser()

    # Allow user to choose a plot type
    plot_options = [
        "Scatterplot Matrix (Original Data)",
        "Parallel Coordinates Plot (Original Data)",
        "3D Scatter Plot (Reduced Data)",
        "Heatmap (Correlation Matrix)"
    ]
    selected_plot = st.sidebar.selectbox("Select a plot", plot_options)

    if selected_plot == "Scatterplot Matrix (Original Data)":
        st.subheader("Scatterplot Matrix of Original Data")
        # Assume the function returns a figure
        fig = visualiser.scatterplot_matrix(data, hue="target")
        st.pyplot(fig)

    elif selected_plot == "Parallel Coordinates Plot (Original Data)":
        st.subheader("Parallel Coordinates Plot of Original Data")
        fig = visualiser.parallel_coordinates_plot(data, class_column="target")
        st.pyplot(fig)

    elif selected_plot == "3D Scatter Plot (Reduced Data)":
        if reduced_data is not None:
            st.subheader("3D Scatter Plot of Reduced Data")
            fig = visualiser.scatter_3d(reduced_data, x="PC1", y="PC2", z="target", hue="target")
            st.pyplot(fig)
        else:
            st.info("Reduced data is not available. Please perform dimensionality reduction on the main page.")
        
    elif selected_plot == "Heatmap (Correlation Matrix)":
        st.subheader("Heatmap of Correlation Matrix")
        fig = visualiser.heatmap(data.corr())
        st.pyplot(fig)

