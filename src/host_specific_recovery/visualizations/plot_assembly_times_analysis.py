import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_characteristic_time(outputs, color, marker, label, margin, fig_path, fig_name, save_fig=False):
    """
    Plot characteristic times of returned vs new species.
    :param outputs: Dictionary containing 'char_time_returned' and 'char_time_new' arrays
    :param color: Color for the scatter points
    :param marker: Marker style for the scatter points
    :param label: Label for the scatter points
    :param margin: Margin to extend the axes limits
    :param fig_path: path to save the figure
    :param fig_name: name of the figure
    :param save_fig: Boolean indicating whether to save the figure
    :return: fig, ax
    """

    char_time_returned_dict = outputs["returned_characteristic_time_dict"]
    char_time_new_2_dict = outputs["new_characteristic_time_dict"]
    ids = outputs["keys"]

    char_time_returned = np.array([char_time_returned_dict[key] for key in ids])
    char_time_new = np.array([char_time_new_2_dict[key] for key in ids])

    fig, ax = plt.subplots(figsize=(30, 30))

    # Scatter plots
    ax.scatter(char_time_returned, char_time_new, s=60 ** 2, color=color, marker=marker,
               alpha=0.7, label=label)

    maximal = np.max(np.hstack([char_time_returned, char_time_new]))
    minimal = np.min(np.hstack([char_time_returned, char_time_new]))

    ax.plot([minimal - margin, maximal + margin], [minimal - margin, maximal + margin], color='black', linestyle='--',
            linewidth=8)

    ax.set_xlabel(r'$\mathit{Returned}$ $\mathit{species}$ average time [days]', fontsize=90, labelpad=40)
    ax.set_ylabel(r'$\mathit{New}$ $\mathit{species}$ average time [days]', fontsize=90, labelpad=40)

    ax.tick_params(axis='both', which='major', labelsize=70, width=8, color='black')

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(8)
        ax.spines[spine].set_color('black')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.grid(False)

    ax.set_ylim(minimal - margin, maximal + margin)
    ax.set_xlim(minimal - margin, maximal + margin)

    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path + fig_name, dpi=300)
    else:
        plt.show()
    return fig, ax

def plot_bars(outputs, lables=None, index=0):

    new_counts = outputs["new_counts"]
    returned_counts = outputs["returned_counts"]
    contingency_tables = outputs["contingency_tables"]

    if lables is None:
        lables = list(contingency_tables[next(iter(contingency_tables))].keys())

    nc = new_counts[index]
    rc = returned_counts[index]

    fig = make_subplots(rows=2, cols=1, subplot_titles=('New species', 'Returned species'),
                        vertical_spacing=0.35)

    bar_new = go.Bar(x=lables, y=nc, name='New species', marker=dict(color='#DC3912'))

    bar_returned = go.Bar(x=lables, y=rc, name='Returned species', marker=dict(color='#3366CC'))

    fig.add_trace(bar_new, row=1, col=1)
    fig.add_trace(bar_returned, row=2, col=1)

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=150)

    fig.update_layout(
        height=2000,
        width=2000,
        showlegend=False,
        yaxis_title='Counts',
        yaxis2_title='Counts',
        xaxis_title='Day',
        xaxis2_title='Day',
        font=dict(color='black'),
        plot_bgcolor='white',
        margin=dict(
            l=100,
            r=100,
            t=200,
            b=100
        ),
        xaxis=dict(showline=True, linewidth=10, linecolor='black', showgrid=False, title_font=dict(size=100),
                   tickfont=dict(size=80), title_standoff=60),
        xaxis2=dict(showline=True, linewidth=10, linecolor='black', showgrid=False, title_font=dict(size=100),
                    tickfont=dict(size=80), title_standoff=60),
        yaxis=dict(showline=True, linewidth=10, linecolor='black', showgrid=False, title_font=dict(size=100),
                   tickfont=dict(size=70), title_standoff=60),
        yaxis2=dict(showline=True, linewidth=10, linecolor='black', showgrid=False, title_font=dict(size=100),
                    tickfont=dict(size=70), title_standoff=60),
    )
    return fig
