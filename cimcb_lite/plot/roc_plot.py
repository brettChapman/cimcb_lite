import numpy as np
from bokeh.models import Band, HoverTool 
from bokeh.plotting import ColumnDataSource, figure
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample


def roc_plot(fpr, tpr, tpr_ci, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", title="", errorbar=False):
    """Creates a rocplot using Bokeh."""
    
    # Get CI
    tpr_lowci = tpr_ci[0]
    tpr_uppci = tpr_ci[1]
    auc = metrics.auc(fpr, tpr)
    
    # specificity and ci-interval for HoverTool
    spec = 1 - fpr
    ci = (tpr_uppci - tpr_lowci) / 2

    # Figure
    data = {"x": fpr, "y": tpr, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec":spec, "ci": ci}
    source = ColumnDataSource(data=data)
    fig = figure(title=title, plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))
    
    # Figure: add line
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal distribution line")
    figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="ROC Curve (Train)", source=source)
    fig.add_tools(HoverTool(renderers=[figline],tooltips=[("Specificity", "@spec{1.111}"),("Sensitivity", "@y{1.111} (+/- @ci{1.111})"),],))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
    fig.add_layout(figband)
    
    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin() # this find the closest value in fpr to errorbar fpr
        fpr_eb = fpr[idx] 
        tpr_eb = tpr[idx]
        tpr_lowci_eb = tpr_lowci[idx]
        tpr_uppci_eb = tpr_uppci[idx]
        roc_whisker_line = fig.multi_line([[fpr_eb, fpr_eb]], [[tpr_lowci_eb, tpr_uppci_eb]], line_alpha=1, line_color="black")
        roc_whisker_bot = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_lowci_eb, tpr_lowci_eb]], line_color="black")
        roc_whisker_top = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_uppci_eb, tpr_uppci_eb]], line_alpha=1, line_color="black")
        fig.circle([fpr_eb], [tpr_eb], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white",)  

    # Change font size
    fig.title.text_font_size = "11pt"
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size
    fig.legend.label_text_font = "10pt"

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Edit legend
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10pt"
    if legend is False:
        fig.legend.visible = False
    return fig

