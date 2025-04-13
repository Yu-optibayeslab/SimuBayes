# visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import parallel_coordinates, radviz

class Visualiser:
    def __init__(self):
        pass

    def parallel_coordinates_plot(self, data, class_column=None):
        # Create a figure and axes using subplots
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot using the provided ax
        parallel_coordinates(data, class_column=class_column, colormap="viridis", ax=ax)
        ax.set_title("Parallel Coordinates Plot")
        return fig

    def scatterplot_matrix(self, data, hue=None):
        # seaborn.pairplot returns a PairGrid object
        g = sns.pairplot(data, hue=hue, palette="viridis")
        # Set a super title for the entire grid
        g.fig.suptitle("Scatterplot Matrix", y=1.02)
        return g.fig

    def radviz(self, data, class_column):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Use the provided ax in radviz function
        radviz(data, class_column=class_column, colormap="viridis", ax=ax)
        ax.set_title("Radial Visualization (RadViz)")
        return fig

    def heatmap(self, data, annot=True, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, annot=annot, cmap=cmap, ax=ax)
        ax.set_title("Heatmap")
        return fig

    def scatter_2d(self, data, x, y, z, s):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # Use data[hue] if hue is provided, else None
        scatter = ax.scatter(data[x], data[y], c=data[z], cmap="viridis", s=s)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("2D Scatter Plot")
        return fig

    def scatter_3d(self, data, x, y, z, hue=None):
        """
        Create an interactive 3D scatter plot using Plotly.
    
        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            z (str): The column name for the z-axis.
            hue (str, optional): The column name for color coding. Defaults to None.
    
        Returns:
            plotly.graph_objects.Figure: The interactive 3D scatter plot.
        """
        # Create the 3D scatter plot
        if hue:
            fig = px.scatter_3d(data, x=x, y=y, z=z, color=hue, size_max=10, size=[5] * len(data), color_continuous_scale='viridis')
        else:
            fig = px.scatter_3d(data, x=x, y=y, z=z, size_max=10, size=[5] * len(data))
    
        # Update layout for better visualization
        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            ),
            margin=dict(l=0, r=0, b=0, t=30)  # Adjust margins
        )
        return fig

'''
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            # Use data[hue] if hue is provided, else None
            scatter = ax.scatter(data[x], data[y], data[z], c=data[hue] if hue else None, cmap="viridis")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            if hue:
                fig.colorbar(scatter, ax=ax, label=hue)
            ax.set_title("3D Scatter Plot")
            return fig

'''
