import matplotlib.pyplot as plt


def plot_survived_species_analysis(outputs, x_vals, x_labels, dir=None):
    results_matrix = 10 ** outputs["results_matrix"]
    mean = 10 ** outputs["mean"]

    fig, ax = plt.subplots(figsize=(25, 25))

    for row in results_matrix:
        ax.plot(x_vals, row[0:-1], linewidth=15, color='#7f7f7f')

        ax.plot(x_vals, mean[0:-1], linewidth=30, color='black')

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=100, rotation=-90, color="black")
        ax.tick_params(axis='x', width=10, colors='black')

        ax.set_yscale('log')
        ax.set_ylabel("Relative abundance", fontsize=150, labelpad=60, color="black")
        ax.tick_params(axis='y', labelsize=120, width=10, colors='black')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0e}"))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(10)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(10)
        ax.spines['left'].set_color('black')

        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        fig.tight_layout()

        if dir is not None:
            plt.savefig(dir, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.show()
