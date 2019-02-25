import numpy as np
from bokeh.models import Band, HoverTool 
from bokeh.plotting import ColumnDataSource, figure
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample


def roc(Y_pred, Y, nboot=1000, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt"):
    """Creates a rocplot using Bokeh."""
    
    # Set bootstrap indices
    bootsam_Y = []
    bootsam_ypred = []
    for i in range(nboot):
        Y_res, ypred_res = resample(Y, Y_pred)
        bootsam_Y.append(Y_res)
        bootsam_ypred.append(ypred_res)
    
    # Calculate tpr, auc for each nboot over linspace
    tprs = []
    aucs = []
    fpr_linspace = np.linspace(0, 1, 1000)  
    for i in range(nboot):
        fpr, tpr, threshold = metrics.roc_curve(bootsam_Y[i], bootsam_ypred[i], pos_label=1)
        tprs.append(interp(fpr_linspace, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
    
    # Get tpr (lowci, median, upperci)
    tpr_lowci = np.percentile(tprs, 2.5, axis=0)
    tpr_median = np.percentile(tprs, 50, axis=0)
    tpr_uppci = np.percentile(tprs, 97.5, axis=0)
    auc_median = np.percentile(aucs, 50, axis=0)
    spec = 1 - fpr_linspace
    ci = tpr_median - tpr_lowci

    # Figure
    data = {"x": fpr_linspace, "y": tpr_median, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec": spec, "ci": ci}
    source = ColumnDataSource(data=data)
    fig = figure(plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))
    
    # Figure: add line
    figline = fig.line("x", "y", color="green", line_width=4, alpha=0.6, legend="ROC Curve (AUC = {:0.2f})".format(auc_median), source=source)
    fig.add_tools(HoverTool(renderers=[figline],tooltips=[("Specificity", "@spec{1.111}"),("Sensitivity", "@y{1.111} (+/- @ci{1.111})"),],))
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal distribution line",)

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
    fig.add_layout(figband)

    # Change font size
    fig.title.text_font_size = "16pt"
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

