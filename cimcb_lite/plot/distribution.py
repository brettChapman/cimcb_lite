import numpy as np
from scipy import stats
from bokeh.plotting import figure


def distribution(X, group, bins=50, hist=True, kde=True, title="Density Plot", xlabel="x", ylabel="Pr(x)", font_size="20pt", label_font_size="13pt", width=500, height=400, color_hist="green", color_kde="mediumturquoise"):
    """Creates a distribution plot using Bokeh."""
    
    # Split into 2 groups 
    group_unique = np.sort(np.unique(group)) 
    x1 = X[group == group_unique[0]] 
    x2 = X[group != group_unique[0]]

    # Density curve
    x1_min, x1_max = x1.min(), x1.max()
    x1_padding = (x1_max - x1_min) * 0.5
    x1_grid = np.linspace(x1_min - x1_padding, x1_max + x1_padding, 500)
    x1_pdf = stats.gaussian_kde(x1, "scott")
    x1_pdf_grid = x1_pdf(x1_grid)

    # Density curve
    x2_min, x2_max = x2.min(), x2.max()
    x2_padding = (x2_max - x2_min) * 0.5
    x2_grid = np.linspace(x2_min - x2_padding, x2_max + x2_padding, 500)
    x2_pdf = stats.gaussian_kde(x2, "scott")
    x2_pdf_grid = x2_pdf(x2_grid)

    max_val_a = max(abs(max(x1_grid)), abs(min(x1_grid)))
    max_val_b = max(abs(max(x2_grid)), abs(min(x2_grid)))
    max_val_final = 0.1 * max(max_val_a, max_val_b)
    x_range_min = min(min(x1_grid) - max_val_final, min(x2_grid) - max_val_final)
    x_range_max = max(max(x1_grid) + max_val_final, max(x2_grid) + max_val_final)
    new_x_range = (x_range_min, x_range_max)

    # Figure
    fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, plot_width=width, plot_height=height, x_range=new_x_range, y_range=(0, max(max(x1_pdf_grid) * 1.1, max(x2_pdf_grid) * 1.1)))
    if hist is True:
        fig.quad(top=hists, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color_hist, line_color="white", alpha=0.5)
    if kde is True:
        fig.patch(x1_grid, x1_pdf_grid, alpha=0.3, color="red", line_color="grey", line_width=1)
        fig.patch(x2_grid, x2_pdf_grid, alpha=0.3, color="blue", line_color="grey", line_width=1)

    # Y-axis should always start at 0
    fig.y_range.start = 0

    # Font-sizes
    fig.title.text_font_size = font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    return fig
