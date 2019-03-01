import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from bokeh.layouts import gridplot 
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import Circle, HoverTool, TapTool
from tqdm import tqdm
from bokeh.plotting import output_notebook, show 
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics


class kfold(BaseCrossVal):
    """ Exhaustitive search over param_dict calculating binary metrics. 
    
    Parameters
    ----------
    model : object
        This object is assumed to store bootlist attributes in .model (e.g. modelPLS.model.x_scores_).
    
    X : array-like, shape = [n_samples, n_features]
        Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.
    
    Y : array-like, shape = [n_samples, 1]
        Response variables, where n_samples is the number of samples.
        
    param_dict : dict
        List of attributes to calculate and return bootstrap confidence intervals.
    
    folds: : a positive integer, (default 10)
        The number of folds used in the computation.

    
    bootnum : a positive integer, (default 100)
        The number of bootstrap samples used in the computation for the plot. 
 

    Attributes
    -------
    Run: Runs all functions prior to plot.
    
    Plot: Creates a R2/Q2 plot.
    
  
    """
    
    def __init__(self, model, X, Y, param_dict, folds=10, bootnum=100):
        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds, bootnum=bootnum)
        self.crossval_idx = StratifiedKFold(n_splits=folds)
        
    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""
        self.ypred_full = []
        self.ypred_cv = []
        for params in self.param_list:
            # Set hyper-parameters
            params_i = params
            model_i = self.model(**params_i)
            # Full
            model_i.train(self.X, self.Y)
            ypred_full_i = model_i.test(self.X)
            self.ypred_full.append(ypred_full_i)
            # CV (for each fold)
            ypred_cv_i = self._calc_cv_ypred(model_i, self.X, self.Y)
            self.ypred_cv.append(ypred_cv_i)

    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        stats_list = []
        self.full_boot_metrics = []
        self.cv_boot_metrics = []
        for i in range(len(self.param_list)):
            # Create dictionaries with binary_metrics
            stats_full_i = binary_metrics(self.Y, self.ypred_full[i])
            stats_cv_i = binary_metrics(self.Y, self.ypred_cv[i])
            # Add _cv to all metrics in stats_cv and change R2_cv to Q2
            stats_cv_i = {f"{k}_cv": v for k, v in stats_cv_i.items()}
            stats_cv_i["Q2"] = stats_cv_i.pop("R2_cv")
            # Combine and append
            stats_combined = {**stats_full_i, **stats_cv_i}
            stats_list.append(stats_combined)
        self.table = self._format_table(stats_list) #Transpose, Add headers
        return self.table
    
    def run(self):
        """Runs all functions prior to plot."""
        self.calc_ypred()
        self.calc_stats()
        if self.bootnum > 1:
            self.calc_ypred_boot()
            self.calc_stats_boot()
            
    def calc_ypred_boot(self):
        """Calculates ypred full and ypred cv for each bootstrap resample."""
        self.ytrue_boot = []
        self.ypred_full_boot = []
        self.ypred_cv_boot = []
        for i in tqdm(range(self.bootnum), desc= "Kfold"):
            bootidx_i = np.random.choice(len(self.Y), len(self.Y))
            newX = self.X[bootidx_i, :]
            newY = self.Y[bootidx_i]
            ypred_full_nboot_i = []
            ypred_cv_nboot_i = []
            for params in self.param_list:
                # Set hyper-parameters
                model_i = self.model(**params)
                # Full
                model_i.train(newX, newY)
                ypred_full_i = model_i.test(newX)
                ypred_full_nboot_i.append(ypred_full_i)
                # cv
                ypred_cv_i = self._calc_cv_ypred(model_i, newX, newY)
                ypred_cv_nboot_i.append(ypred_cv_i)
            self.ytrue_boot.append(newY)
            self.ypred_full_boot.append(ypred_full_nboot_i)
            self.ypred_cv_boot.append(ypred_cv_nboot_i)
            
    def calc_stats_boot(self):
        """Calculates binary statistics from ypred full and ypred cv for each bootstrap resample."""
        self.full_boot_metrics = []
        self.cv_boot_metrics = []
        for i in range(len(self.param_list)):
            stats_full_i = []
            stats_cv_i = []
            for j in range(self.bootnum):
                stats_full = binary_metrics(self.ytrue_boot[j], self.ypred_full_boot[j][i])
                stats_full_i.append(stats_full)
                stats_cv = binary_metrics(self.ytrue_boot[j], self.ypred_cv_boot[j][i])
                stats_cv_i.append(stats_cv)
            self.full_boot_metrics.append(stats_full_i)
            self.cv_boot_metrics.append(stats_cv_i)
            
    def _calc_cv_ypred(self, model_i, X, Y):
        """Method used to calculate ypred cv."""
        ypred_cv_i = [None] * len(Y)
        for train, test in self.crossval_idx.split(self.X, self.Y):
            X_train = X[train, :]
            Y_train = Y[train]
            X_test = X[test, :]
            Y_test = Y[test]
            model_i.train(X_train, Y_train)
            ypred_cv_i_j = model_i.test(X_test, Y_test)
            # Return value to y_pred_cv in the correct position # Better way to do this
            for (idx, val) in zip(test, ypred_cv_i_j):
                ypred_cv_i[idx] = val.tolist()
        return ypred_cv_i
    
    def _format_table(self, stats_list):
        """Make stats pretty (pandas table -> proper names in columns)."""
        table = pd.DataFrame(stats_list).T
        param_list_string = []
        for i in range(len(self.param_list)):
            param_list_string.append(str(self.param_list[i]))
        table.columns = param_list_string
        return table

    def plot(self):
        "Create a R2/Q2 plot."
        
        # get r2, q2, and diff
        r2 = self.table.loc["R2"]
        q2 = self.table.loc["Q2"]
        diff = r2 - q2
        
        # round r2, q2, and diff for hovertool
        r2_text = []
        q2_text = []
        diff_text = []
        for j in range(len(r2)):
            r2_text.append("%.2f" % round(r2[j], 2))
            q2_text.append("%.2f" % round(q2[j], 2))
            diff_text.append("%.2f" % round(diff[j], 2))
        
        # get key, values (as string) from param_dict (key -> title, values -> x axis values)
        for k, v in self.param_dict.items():
            key = k
            values = v
        values_string = [str(i) for i in values]
        
        # store data in ColumnDataSource for Bokeh
        data = dict(r2=r2, q2=q2, diff=diff, r2_text=r2_text, q2_text=q2_text, diff_text=diff_text, values_string=values_string)
        source = ColumnDataSource(data=data)
        
        fig1_yrange = (min(diff) - max(0.1 * (min(diff)), 0.03), max(diff) + max(0.1 * (max(diff)), 0.03))
        fig1_xrange = (min(q2) - max(0.1 * (min(q2)), 0.03), max(q2) + max(0.1 * (max(q2)), 0.03))
        
        # Figure 1 (DIFFERENCE (R2 - Q2) vs. Q2)
        fig1 = figure(x_axis_label="Q2", y_axis_label="DIFFERENCE (R2 - Q2)", title="DIFFERENCE (R2 - Q2) vs. Q2", tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", y_range=fig1_yrange, x_range=fig1_xrange,plot_width=480, plot_height=400)
        fig1.title.text_font_size = "14pt"
        
        # Figure 1: Add a line
        fig1_line = fig1.line(q2, diff, line_width=2, line_color="black", line_alpha=0.25)
        
        # Figure 1: Add circles (interactive click)
        fig1_circ = fig1.circle("q2", "diff", size=10, alpha=0.7, color="green", source=source)
        fig1_circ.selection_glyph = Circle(fill_color="green", line_width=2, line_color="black")
        fig1_circ.nonselection_glyph.fill_color = "green"
        fig1_circ.nonselection_glyph.fill_alpha = 0.4
        fig1_circ.nonselection_glyph.line_color = "white"
        
        # Figure 1: Add hovertool
        fig1.add_tools(HoverTool(renderers=[fig1_circ],tooltips=[(key, "@values_string"), ("R2", "@r2_text"), ("Q2", "@q2_text"), ("Diff", "@diff_text"),],))
        
        # Figure 1: Extra formating
        fig1.axis.major_label_text_font_size = "8pt"
        fig1.xaxis.axis_label_text_font_size = "12pt"
        fig1.yaxis.axis_label_text_font_size = "12pt"

        # Figure 2: R2/Q2 
        fig2 = figure(x_axis_label=key, y_axis_label="Value", title="R2/Q2 vs. {}".format(key), plot_width=480, plot_height=400, x_range=pd.unique(values_string), y_range=(0, 1.1), tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")
        fig2.title.text_font_size = "14pt"
        
        # Figure 2: add confidence intervals if bootnum > 1
        if self.bootnum > 1:
            lower_ci_r2 = []
            upper_ci_r2 = []
            lower_ci_q2 = []
            upper_ci_q2 = []
            # Get all upper, lower 95% CI (r2/q2) for each specific n_component and append 
            for m in range(len(self.full_boot_metrics)):
                r2_boot = []
                q2_boot = []
                for k in range(len(self.full_boot_metrics[0])):
                    r2_boot.append(self.full_boot_metrics[m][k]['R2'])
                    q2_boot.append(self.cv_boot_metrics[m][k]['R2']) # Wasn't renamed to Q2
                # Calculated percentile 95% CI and append
                r2_bias = np.mean(r2_boot) - r2[m]
                q2_bias = np.mean(q2_boot) - q2[m]
                lower_ci_r2.append(np.percentile(r2_boot, 2.5) - r2_bias)
                upper_ci_r2.append(np.percentile(r2_boot, 97.5) - r2_bias)
                lower_ci_q2.append(np.percentile(q2_boot, 2.5) - q2_bias)
                upper_ci_q2.append(np.percentile(q2_boot, 97.5) - q2_bias)
            
            # Plot as a patch
            x_patch = np.hstack((values_string, values_string[::-1]))
            y_patch_r2 = np.hstack((lower_ci_r2, upper_ci_r2[::-1]))
            fig2.patch(x_patch, y_patch_r2, alpha=0.10, color="red")
            y_patch_q2 = np.hstack((lower_ci_q2, upper_ci_q2[::-1]))
            fig2.patch(x_patch, y_patch_q2, alpha=0.10, color="blue")
        
        # Figure 2: add r2
        fig2_line_r2 = fig2.line(values_string, r2, line_color="red", line_width=2)
        fig2_circ_r2 = fig2.circle("values_string", "r2", line_color="red", fill_color="white", fill_alpha=1, size=8, source=source)
        fig2_circ_r2.selection_glyph = Circle(line_color="red", fill_color="white", line_width=2)
        fig2_circ_r2.nonselection_glyph.line_color = "red"
        fig2_circ_r2.nonselection_glyph.fill_color = "white"
        fig2_circ_r2.nonselection_glyph.line_alpha = 0.4
        
        # Figure 2: add q2
        fig2_line_q2 = fig2.line(values_string, q2, line_color="blue", line_width=2)
        fig2_circ_q2 = fig2.circle("values_string", "q2", line_color="blue", fill_color="white", fill_alpha=1, size=8, source=source)
        fig2_circ_q2.selection_glyph = Circle(line_color="blue", fill_color="white", line_width=2)
        fig2_circ_q2.nonselection_glyph.line_color = "blue"
        fig2_circ_q2.nonselection_glyph.fill_color = "white"
        fig2_circ_q2.nonselection_glyph.line_alpha = 0.4
        
        # Add hovertool and taptool
        fig2.add_tools(HoverTool(renderers=[fig2_circ_r2], tooltips=[("R2", "@r2")], mode="vline"))
        fig2.add_tools(HoverTool(renderers=[fig2_circ_q2], tooltips=[("Q2", "@q2")], mode="vline"))
        fig2.add_tools(TapTool(renderers=[fig2_circ_r2, fig2_circ_q2]))

        # Create a grid and output figures
        grid = np.full((1, 2), None) 
        grid[0, 0] = fig1
        grid[0, 1] = fig2
        fig = gridplot(grid.tolist(), merge_tools=True)
        output_notebook()
        show(fig)
    

