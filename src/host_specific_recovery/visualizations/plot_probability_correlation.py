from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from plotly import graph_objects as go

def plot_probability_correlation(prob_df, min_counts, min_counts_ret):

    condition = (prob_df["counts"] > min_counts) & (prob_df["counts returned"] > min_counts_ret)

    x = prob_df[condition]["ps"].to_numpy()
    y = prob_df[condition]["t_mean"].to_numpy()

    spearman_corr, spearman_p = spearmanr(x, y)
    pearson_corr, pearson_p = pearsonr(x, y)

    print(f"Spearman Correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4g}")
    print(f"Pearson Correlation: {pearson_corr:.4f}, p-value: {pearson_p:.4g}")
    print(" ")

    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    Y_pred = reg.predict(X)

    # Plots
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#109618', size=60)))

    fig.add_trace(go.Scatter(x=x, y=Y_pred.flatten(), mode='lines', line=dict(color='#990099', width=15)))

    fig.update_xaxes(title_text="Probability to survive", title_font=dict(size=150, color="black"),
                     tickfont=dict(size=120, color="black"), title_standoff=80, linecolor="black",
                     tickcolor="black", showline=True, linewidth=10, mirror=False, dtick=0.2)

    fig.update_yaxes(title_text="Average return time [day]", title_font=dict(size=150, color="black"),
                     tickfont=dict(size=120, color="black"), title_standoff=80, linecolor="black", tickcolor="black",
                     showline=True, linewidth=10, mirror=False)

    fig.update_layout(width=2500, height=2500, plot_bgcolor='white', showlegend=False)

    fig.show()
