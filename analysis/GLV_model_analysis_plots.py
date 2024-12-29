from utils.general_functions import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

post_sim = np.load("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim.npy")
post_sim_others = np.load("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim_others.npy")
ABX_sim = np.load("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ABX_sim.npy")
post_sim_off = np.load("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim_off.npy")
post_sim_others_off = np.load("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim_others_off.npy")
ABX_sim_off = np.load("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ABX_sim_off.npy")

iters = 100
sims_new, sims_survived = calc_similarity(post_sim, ABX_sim, post_sim_others, iters)
sims_new_off, sims_survived_off = calc_similarity(post_sim_off, ABX_sim_off, post_sim_others_off, iters)

spearman_corr, spearman_p = spearmanr(sims_survived, sims_new)
pearson_corr, pearson_p = pearsonr(sims_survived, sims_new)

# Print correlation values
print(f"Spearman Correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4g}")
print(f"Pearson Correlation: {pearson_corr:.4f}, p-value: {pearson_p:.4g}")

# Perform linear regression
X = np.array(sims_survived).reshape(-1, 1)
Y = np.array(sims_new).reshape(-1, 1)
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(
    x=sims_survived,
    y=sims_new,
    mode='markers',
    marker=dict(color='blue', size=30),
))

fig.add_trace(go.Scatter(
    x=sims_survived,
    y=Y_pred.flatten(),
    mode='lines',
    line=dict(color='darkblue', width=10),
))

fig.update_xaxes(title_text="Z-score (Initial species)", title_font=dict(size=70), tickfont=dict(size=60),
                 linecolor='black', showline=True, linewidth=4, mirror=False)
fig.update_yaxes(title_text="Z-score (New species)", title_font=dict(size=70), tickfont=dict(size=60),
                 linecolor='black', showline=True, linewidth=4, mirror=False)

fig.update_layout(
    width=1600,
    height=1600,
    plot_bgcolor='white',
    showlegend=False,
)

fig.show()

spearman_corr, spearman_p = spearmanr(sims_survived_off, sims_new_off)
pearson_corr, pearson_p = pearsonr(sims_survived_off, sims_new_off)

# Print correlation values
print(f"Spearman Correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4g}")
print(f"Pearson Correlation: {pearson_corr:.4f}, p-value: {pearson_p:.4g}")

# Perform linear regression
X = np.array(sims_survived_off).reshape(-1, 1)
Y = np.array(sims_new_off).reshape(-1, 1)
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(
    x=sims_survived_off,
    y=sims_new_off,
    mode='markers',
    marker=dict(color='blue', size=30),
))

fig.add_trace(go.Scatter(
    x=sims_survived_off,
    y=Y_pred.flatten(),
    mode='lines',
    line=dict(color='darkblue', width=10),
))

fig.update_xaxes(title_text="Z-score (Initial species)", title_font=dict(size=70), tickfont=dict(size=60),
                 linecolor='black', showline=True, linewidth=4, mirror=False)
fig.update_yaxes(title_text="Z-score (New species)", title_font=dict(size=70), tickfont=dict(size=60),
                 linecolor='black', showline=True, linewidth=4, mirror=False)

fig.update_layout(
    width=1600,
    height=1600,
    plot_bgcolor='white',
    showlegend=False,
)

fig.show()