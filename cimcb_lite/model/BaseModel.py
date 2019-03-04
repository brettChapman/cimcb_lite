from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import scipy
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.models import HoverTool, Band
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.models.annotations import Title
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show 
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample
from ..bootstrap import Perc, BC, BCA
from ..plot import scatter, scatterCI, boxplot, distribution, permutation_test, roc_plot
from ..utils import binary_metrics, roc_calculate


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

    def evaluate(self, testset=None, specificity=False, cutoffscore=False, bootnum=1000):
        """Returns a figure containing a Violin plot, Distribution plot, ROC plot and Binary Metrics statistics."""
        
        Ytrue_train = self.Y
        Yscore_train = self.Y_pred.flatten()
        
        # Get Ytrue_test, Yscore_test from testset
        if testset is not None:
            Ytrue_test = np.array(testset[0])
            Yscore_test = np.array(testset[1])
            
            # Error checking
            if len(Ytrue_test) != len(Yscore_test):
                raise ValueError("evaluate can't be used as length of Ytrue does not match length of Yscore in test set.")
            if len(np.unique(Ytrue_test)) != 2:
                raise ValueError("Ytrue_test needs to have 2 groups. There is {}".format(len(np.unique(Y))))
            if np.sort(np.unique(Ytrue_test))[0] != 0:
                raise ValueError("Ytrue_test should only contain 0s and 1s.")
            if np.sort(np.unique(Ytrue_test))[1] != 1:
                raise ValueError("Ytrue_test should only contain 0s and 1s.")

            # Get Yscore_combined and Ytrue_combined_name (Labeled Ytrue)
            Yscore_combined = np.concatenate([Yscore_train, Yscore_test])
            Ytrue_combined = np.concatenate([Ytrue_train, Ytrue_test + 2]) # Each Ytrue per group is unique
            Ytrue_combined_name = Ytrue_combined.astype(np.str)
            Ytrue_combined_name[Ytrue_combined == 0] = "Train (0)"
            Ytrue_combined_name[Ytrue_combined == 1] = "Train (1)"
            Ytrue_combined_name[Ytrue_combined == 2] = "Test (0)"
            Ytrue_combined_name[Ytrue_combined == 3] = "Test (1)"
        
        # Expliclity states which metric and value is used for the error_bar
        if specificity is not False:
            metric = 'specificity'
            val = specificity
        elif cutoffscore is not False:
            metric = 'cutoffscore'
            val = cutoffscore
        else:
            metric = 'specificity'
            val = 0.8
            
        # ROC plot
        tpr, fpr, tpr_ci, stats, stats_bootci = roc_calculate(Ytrue_train, Yscore_train, bootnum=100, metric=metric, val=val)
        roc_title = "Specificity: {}".format(np.round(stats['val_specificity'],2))
        roc_bokeh = roc_plot(tpr, fpr, tpr_ci, width=320, height=315, title=roc_title, errorbar=stats['val_specificity'])
        if testset is not None:
            fpr_test, tpr_test, threshold_test = metrics.roc_curve(Ytrue_test, Yscore_test, pos_label=1, drop_intermediate=False)
            fpr_test = np.insert(fpr_test, 0, 0)
            tpr_test = np.insert(tpr_test, 0, 0)
            roc_bokeh.line(fpr_test, tpr_test, color="red", line_width=3.5, alpha=0.6, legend="ROC Curve (Test)") # Add ROC Curve(Test) to roc_bokeh

        # Violin plot
        if testset is None:
            violin_bokeh = boxplot(Yscore_train, Ytrue_train, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315)
        else:
            violin_bokeh = boxplot(Yscore_combined, Ytrue_combined_name, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#fcaeae", "#aed3f9","#FFCCCC", "#CCE5FF"], width=320, height=315, group_name=['Train (0)', 'Test (0)', 'Train (1)', 'Test (1)'], group_name_sort=['Test (0)', 'Test (1)', 'Train (0)', 'Train (1)'])
        violin_bokeh.multi_line([[-100, 100]], [[stats['val_cutoffscore'], stats['val_cutoffscore']]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Distribution plot
        if testset is None:
            dist_bokeh = distribution(Yscore_train, group=Ytrue_train, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)
        else:
            dist_bokeh = distribution(Yscore_combined, group=Ytrue_combined_name, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)
        dist_bokeh.multi_line([[stats['val_cutoffscore'], stats['val_cutoffscore']]], [[-100, 100]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")
        
        # Man-Whitney U for Table (round and use scienitic notation if p-value > 0.001)
        manw_pval = scipy.stats.mannwhitneyu(Yscore_train[Ytrue_train==0], Yscore_train[Ytrue_train==1])[1]
        if manw_pval > 0.001:
            manw_pval_round = "%0.2f" % manw_pval
        else:
            manw_pval_round = "%0.2e" % manw_pval
        if testset is not None:
            testmanw_pval = scipy.stats.mannwhitneyu(Yscore_test[Ytrue_test==0], Yscore_test[Ytrue_test==1])[1]
            if testmanw_pval > 0.001:
                testmanw_pval_round = "%0.2f" % testmanw_pval
            else:
                testmanw_pval_round = "%0.2e" % testmanw_pval
        
        # Create a stats table for test 
        if testset is not None:
            teststats = binary_metrics(Ytrue_test, Yscore_test, cut_off=stats['val_cutoffscore'])
            teststats_round = {}
            for i in teststats.keys():
                teststats_round[i] = np.round(teststats[i],2)
            
        # Round stats, and stats_bootci for Table
        stats_round = {} 
        for i in stats.keys():
            stats_round[i] = np.round(stats[i],2)
        bootci_round = {} 
        for i in stats_bootci.keys():
            bootci_round[i] = np.round(stats_bootci[i],2)
        
        # Create table 
        tabledata = dict(evaluate=[["Train"]],
                         manw_pval=[["{}".format(manw_pval_round)]],
                         auc=[["{} ({}, {})".format(stats_round['auc'], bootci_round['auc'][0], bootci_round['auc'][1])]],
                         accuracy=[["{} ({}, {})".format(stats_round['accuracy'], bootci_round['accuracy'][0], bootci_round['accuracy'][1])]],
                         precision=[["{} ({}, {})".format(stats_round['precision'], bootci_round['precision'][0], bootci_round['precision'][1])]],
                         sensitivity=[["{} ({}, {})".format(stats_round['sensitivity'], bootci_round['sensitivity'][0], bootci_round['sensitivity'][1])]],
                         specificity=[["{} ({}, {})".format(stats_round['specificity'], bootci_round['specificity'][0], bootci_round['specificity'][1])]],
                         F1score=[["{} ({}, {})".format(stats_round['F1score'], bootci_round['F1score'][0], bootci_round['F1score'][1])]],
                         R2=[["{} ({}, {})".format(stats_round['R2'], bootci_round['R2'][0], bootci_round['R2'][1])]])
        
        # Append test data 
        if testset is not None:
            tabledata['evaluate'].append(["Test"])
            tabledata['manw_pval'].append([testmanw_pval_round]) 
            tabledata['auc'].append([teststats_round['auc']])
            tabledata['accuracy'].append([teststats_round['accuracy']])
            tabledata['precision'].append([teststats_round['precision']])
            tabledata['sensitivity'].append([teststats_round['sensitivity']])
            tabledata['specificity'].append([teststats_round['specificity']])
            tabledata['F1score'].append([teststats_round['F1score']])
            tabledata['R2'].append([teststats_round['R2']])
        
        # Plot table
        source = ColumnDataSource(data=tabledata)
        columns = [
            TableColumn(field="evaluate", title="Evaluate"),
            TableColumn(field="manw_pval", title="MW-U Pvalue"),
            TableColumn(field="R2", title="R2"),
            TableColumn(field="auc", title="AUC"),
            TableColumn(field="accuracy", title="Accuracy"),
            TableColumn(field="precision", title="Precision"),
            TableColumn(field="sensitivity", title="Sensitivity"),
            TableColumn(field="F1score", title="F1score")]
        table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=80)
        
        # Title 
        if specificity is not False:
            title = "Specificity set to: {}".format(np.round(val, 2))
        elif cutoffscore is not False:
            title = "Score cut-off set to: {}".format(np.round(val, 2))
        else:
            title = "Specificity set to: {}".format(np.round(val, 2))
        title_bokeh = "<h3>{}</h3>".format(title)

        # Combine table, violin plot and roc plot into one figure
        fig = gridplot([[violin_bokeh, dist_bokeh, roc_bokeh], [table_bokeh]], toolbar_location="right")
        output_notebook()
        show(column(Div(text=title_bokeh, width=900, height=50), fig))

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
        
    @staticmethod
    def _get_sens_spec(Ytrue, Yscore, cuttoff_val):
        Yscore_round = np.where(np.array(Yscore) > cuttoff_val, 1, 0) 
        tn, fp, fn, tp = metrics.confusion_matrix(Ytrue, Yscore_round).ravel()
        sensitivity  = tp / (tp+fn)
        specificity  = tn / (tn+fp)
        return sensitivity, specificity
    
    @staticmethod
    def _get_sens_cuttoff(Ytrue, Yscore, specificity_val):
        fpr0 = 1- specificity_val
        fpr, sensitivity , thresholds = metrics.roc_curve(Ytrue, Yscore, pos_label=1)
        idx = np.abs(fpr - fpr0).argmin() # this find the closest value in fpr to fpr0
        return sensitivity[idx], thresholds[idx]
    
    @staticmethod
    def _get_sens_spec_cuttoff(Ytrue, Yscore):
        fpr, sensitivity , thresholds = metrics.roc_curve(Ytrue, Yscore, pos_label=1)
        idx = (1 - fpr + sensitivity).argmax() # Using youden index, find max value
        specificity_idx = 1 - fpr[idx]
        return specificity_idx, sensitivity[idx], thresholds[idx]