import numpy as np
from scipy.stats import linregress, pearsonr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_similarity_analysis(outputs: dict):

    unweighted_similarity_vector = outputs["unweighted_similarity_vector"]
    weighted_similarity_vector = outputs["weighted_similarity_vector"]

    r, p = pearsonr(np.array(unweighted_similarity_vector), np.array(weighted_similarity_vector))

    print(f"Pearson correlation: {r}, p-value: {p}")

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(unweighted_similarity_vector, weighted_similarity_vector)
    line_x = np.linspace(min(unweighted_similarity_vector), max(unweighted_similarity_vector), 100)
    line_y = slope * line_x + intercept

    # Plot
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(x=unweighted_similarity_vector, y=weighted_similarity_vector, mode='markers',
                             marker=dict(color='black', size=100)))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='darkred', width=25)))

    fig.update_xaxes(title_text="Unweighted similarity", title_standoff=100, title_font=dict(size=200, color='black'),
                     tickfont=dict(size=150, color='black'), linecolor='black', showline=True, linewidth=10,
                     mirror=False)

    fig.update_yaxes(title_text="Weighted similarity", title_standoff=100, title_font=dict(size=200, color='black'),
                     tickfont=dict(size=150, color='black'), linecolor='black', showline=True, linewidth=10,
                     mirror=False)

    fig.update_layout(width=3200, height=3200, plot_bgcolor='white', showlegend=False)

    fig.show()
