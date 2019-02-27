from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.models import HoverTool, Band
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.models.annotations import Title
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show 
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample
from ..bootstrap import Perc, BC, BCA
from ..plot import scatter, scatterCI, boxplot, distribution, permutation_test
from ..utils import binary_metrics


class BaseModel(ABC):
    """Base class for models: PLS_SIMPLS."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """Trains the model."""
        pass

    @abstractmethod
    def test(self):
        """Tests the model."""
        pass

    @abstractproperty
    def bootlist(self):
        """A list of attributes for bootstrap resampling."""
        pass

    def evaluate(self, evals='default', specificity=False, cutoffscore=False, bootnum=100):
        """Returns a figure containing a Violin plot, Distribution plot, ROC plot and Binary Metrics statistics."""
        
        # if evals is 'default', set Y_pred and Y_true based on train method
        if evals is 'default':
            Y_true = self.Y
            Y_pred = self.Y_pred.flatten()
        else:
            Y_true = np.array(evals[0])
            Y_pred = np.array(evals[1])
            if len(np.unique(Y_true)) != 2:
                raise ValueError("evalute can't be used as Y_true is not binary.")
            if len(Y_true) != len(Y_pred):
                raise ValueError("evaluate can't be used as length of Y_true does not match length of Y_pred.")

        # Violin plot
        violin_bokeh = boxplot(Y_pred, Y_true, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315)

        # Distribution plot
        dist_bokeh = distribution(Y_pred, group=Y_true, hist=False, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)

        ## ROC plot
        # Set bootstrap indices
        bootsam_Y = []
        bootsam_ypred = []
        for i in range(bootnum):
            Y_res, ypred_res = resample(Y_true, Y_pred)
            bootsam_Y.append(Y_res)
            bootsam_ypred.append(ypred_res)

        # Calculate tpr, auc for each nboot over linspace
        tprs = []
        aucs = []
        thresholds = []
        fpr_linspace = np.linspace(0, 1, 1000)
        for i in range(bootnum):
            fpr, tpr, threshold = metrics.roc_curve(bootsam_Y[i], bootsam_ypred[i], pos_label=1)
            tprs.append(interp(fpr_linspace, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            thresholds.append(interp(fpr_linspace, fpr, threshold))
            thresholds[-1][0] = 0.0

        # Get tpr (lowci, median, upperci)
        tpr_lowci = np.percentile(tprs, 2.5, axis=0)
        tpr_median = np.percentile(tprs, 50, axis=0)
        tpr_uppci = np.percentile(tprs, 97.5, axis=0)
        auc_median = np.percentile(aucs, 50, axis=0)
        thresholds_median = np.percentile(thresholds, 50, axis=0)
        spec = 1 - fpr_linspace
        ci = tpr_median - tpr_lowci

        # Plot ROC
        # Figure
        data = {"x": fpr_linspace, "y": tpr_median, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec": spec, "ci": ci}
        source = ColumnDataSource(data=data)
        roc_bokeh = figure(plot_width=320, plot_height=315, x_axis_label="1-Specificity", y_axis_label="Sensitivity", x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))

        # Figure: add line
        figline = roc_bokeh.line("x", "y", color="green", line_width=4, alpha=0.6, legend="ROC Curve (AUC = {:0.2f})".format(auc_median), source=source)
        roc_bokeh.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"),],))
        roc_bokeh.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5,legend="Equal distribution line")

        # Figure: add 95CI band
        figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
        roc_bokeh.add_layout(figband)

        # Change font size
        roc_bokeh.title.text_font_size = "16pt"
        roc_bokeh.xaxis.axis_label_text_font_size = "13pt"
        roc_bokeh.yaxis.axis_label_text_font_size = "13pt"
        roc_bokeh.legend.label_text_font = "10pt"

        # Extra padding
        roc_bokeh.min_border_left = 20
        roc_bokeh.min_border_right = 20
        roc_bokeh.min_border_top = 20
        roc_bokeh.min_border_bottom = 20

        # Edit legend
        roc_bokeh.legend.location = "bottom_right"
        roc_bokeh.legend.label_text_font_size = "10pt"

        # Get metrics (boot)
        if specificity is not False:
            idx = (np.abs(spec - specificity)).argmin()
        elif cutoffscore is not False:
            idx = (np.abs(thresholds_median - cutoffscore)).argmin()
        else:
            idx = (spec + tpr_median - 1).argmax()

        stats = []
        cutoff = []
        for i in range(bootnum):
            cutoff_boot = thresholds[i][idx]
            stats_boot = binary_metrics(bootsam_Y[i], bootsam_ypred[i], cut_off=cutoff_boot)
            stats.append(stats_boot)
            cutoff.append(cutoff_boot)

        # Get plot_cutoff for violin and dist plot
        if specificity is not False:
            plot_cutoff = np.mean(cutoff)
        elif cutoff is not False:
            plot_cutoff = cutoffscore
        else:
            plot_cutoff = thresholds_median[(spec + tpr_median - 1).argmax()]

        # Violin plot line
        violin_bokeh.multi_line([[-100, 100]], [[plot_cutoff, plot_cutoff]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Dist plot line
        dist_bokeh.multi_line([[plot_cutoff, plot_cutoff]], [[-100, 100]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # ROC plot error bar
        plot_spec = fpr_linspace[idx]
        plot_sens = tpr_median[idx]
        plot_sens_low = tpr_lowci[idx]
        plot_sens_upp = tpr_uppci[idx]
        roc_whisker_line = roc_bokeh.multi_line([[plot_spec, plot_spec]], [[plot_sens_low, plot_sens_upp]], line_alpha=1, line_color="black")
        roc_whisker_bot = roc_bokeh.multi_line([[plot_spec - 0.03, plot_spec + 0.03]], [[plot_sens_upp, plot_sens_upp]], line_color="black")
        roc_whisker_top = roc_bokeh.multi_line([[plot_spec - 0.03, plot_spec + 0.03]], [[plot_sens_low, plot_sens_low]], line_alpha=1, line_color="black")
        roc_bokeh.circle([plot_spec], [plot_sens], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white",)

        # Add title to ROC
        t = Title()
        t.text = "Specificity: {}".format(np.round(1 - plot_spec, 2))
        roc_bokeh.title = t

        # Bootstrap to get CI (for table)
        b_auc = []
        b_acc = []
        b_pre = []
        b_sens = []
        b_spec = []
        b_f1 = []
        b_r2 = []
        for i in stats:
            b_auc.append(i["auc"])
            b_acc.append(i["accuracy"])
            b_pre.append(i["precision"])
            b_sens.append(i["sensitivity"])
            b_spec.append(i["specificity"])
            b_f1.append(i["F1score"])
            b_r2.append(i["R2"])

        # Put into table
        source = ColumnDataSource(
            dict(
                auc=tuple([["{} ({}, {})".format(np.round(np.percentile(b_auc,50),2), np.round(np.percentile(b_auc,2.5),2), np.round(np.percentile(b_auc,97.5),2))]]),
                accuracy=tuple([["{} ({}, {})".format(np.round(np.percentile(b_acc,50),2), np.round(np.percentile(b_acc,2.5),2), np.round(np.percentile(b_acc,97.5),2))]]),
                precision=tuple([["{} ({}, {})".format(np.round(np.percentile(b_pre,50),2), np.round(np.percentile(b_pre,2.5),2), np.round(np.percentile(b_pre,97.5),2))]]),
                sensitivity=tuple([["{} ({}, {})".format(np.round(np.percentile(b_sens,50),2), np.round(np.percentile(b_sens,2.5),2), np.round(np.percentile(b_sens,97.5),2))]]),
                specificity=tuple([["{} ({}, {})".format(np.round(np.percentile(b_spec,50),2), np.round(np.percentile(b_spec,2.5),2), np.round(np.percentile(b_spec,97.5),2))]]),
                F1score=tuple([["{} ({}, {})".format(np.round(np.percentile(b_f1,50),2), np.round(np.percentile(b_f1,2.5),2), np.round(np.percentile(b_f1,97.5),))]]),
                R2=tuple([["{} ({}, {})".format(np.round(np.percentile(b_r2,50),2), np.round(np.percentile(b_r2,2.5),2), np.round(np.percentile(b_r2,97.5),2))]]),
            )
        )

        columns = [
            TableColumn(field="R2", title="R2"),
            TableColumn(field="auc", title="auc"),
            TableColumn(field="accuracy", title="accuracy"),
            TableColumn(field="precision", title="precision"),
            TableColumn(field="sensitivity", title="sensitivity"),
            TableColumn(field="F1score", title="F1score"),]

        # Title
        if specificity is not False:
            titleA = "Specificity set to {}".format(specificity)
        elif cutoffscore is not False:
            titleA = "Score cut-off set to: {}".format(cutoffscore)
        else:
            titleA = "Score cut-off set to: {} (using Youden Index)".format(np.round(thresholds_median[idx], 2))
        title = "<h3>{}</h3>".format(titleA)

        # Set table
        table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=70), width=950, height=20)

        # Combine table, violin plot and roc plot into one figure
        fig = gridplot([[violin_bokeh, dist_bokeh, roc_bokeh], [table_bokeh]], toolbar_location="right")
        output_notebook()
        show(column(Div(text=title, width=900, height=20), fig))

    def calc_bootci(self, bootnum=1000, type="perc"):
        """Calculates bootstrap confidence intervals based on bootlist."""
        bootlist = self.bootlist
        if type is "bca":
            boot = BCA(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
        if type is "bc":
            boot = BC(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
        if type is "perc":
            boot = Perc(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
        self.bootci = boot.run()

    def plot_featureimportance(self, PeakTable, peaklist=None, ylabel="Label", sort=True):
        """Plots feature importance metrics."""
        if not hasattr(self, "bootci"):
            print("Use method calc_bootci prior to plot_featureimportance to add 95% confidence intervals to plots.")
            ci_coef = None
            ci_vip = None
        else:
            ci_coef = self.bootci["model.coef_"]
            ci_vip = self.bootci["model.vip_"]

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        # Plot
        fig_1 = scatterCI(self.model.coef_, ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title="Coefficient Plot", sort_abs=sort)
        fig_2 = scatterCI(self.model.vip_, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=1, col_hline=False, title="Variable Importance in Projection (VIP)", sort_abs=sort)
        fig = layout([[fig_1], [fig_2]])
        output_notebook()
        show(fig)

        # Return table with: Idx, Name, Label, Coefficient, 95CI, VIP, 95CI
        coef = pd.DataFrame([self.model.coef_, self.bootci["model.coef_"]]).T
        coef.rename(columns={0: "Coef", 1: "Coef-95CI"}, inplace=True)
        vip = pd.DataFrame([self.model.vip_, self.bootci["model.vip_"]]).T
        vip.rename(columns={0: "VIP", 1: "VIP-95CI"}, inplace=True)

        Peaksheet = PeakTable.copy()
        Peaksheet["Coef"] = coef["Coef"].values
        Peaksheet["Coef-95CI"] = coef["Coef-95CI"].values
        Peaksheet["VIP"] = vip["VIP"].values
        Peaksheet["VIP-95CI"] = vip["VIP-95CI"].values
        return Peaksheet

    def permutation_test(self, nperm=100):
        """Plots permutation test figures."""
        fig = permutation_test(self, self.X, self.Y, nperm=nperm)
        output_notebook()
        show(fig)
