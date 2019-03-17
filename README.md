![image](cimcb_logo.jpg)
### cimcb lite 
cimcb_lite is a lite version of the cimcb package containing the necessary tools for the statistical analysis of untargeted and targeted metabolomics data **(currently for binary classification)**. 

## Installation

### Dependencies
cimcb_lite requires:
- Python (>=3.5)
- Bokeh (>=1.0.0)
- NumPy
- SciPy
- scikit-learn
- Statsmodels
- tqdm

### User installation
The recommend way to install cimcb_lite and dependencies is to using ``conda``:
```console
conda install -c cimcb cimcb_lite
```
or ``pip``:
```console
pip install cimcb_lite
```
Alternatively, to install directly from github:
```console
pip install https://github.com/KevinMMendez/cimcb_lite/archive/master.zip
```

### Quick Start


### Tutorial
Open with Binders:  

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KevinMMendez/BinderTutorial_Workflow/master?filepath=BinderTutorial_Workflow.ipynb)

### API
For futher detail on the usage refer to the docstring.

#### cimcb_lite.model
- [PLS_SIMPLS](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/model/PLS_SIMPLS.py#L14-L36)

#### cimcb_lite.plot
- [boxplot](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/boxplot.py#L8-L18)
- [distribution](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/distribution.py#L6-L16)
- [pca](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/pca.py#L10-L17)
- [permutation_test](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/permutation_test.py#L13-L27)
- [roc_plot](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/roc.py#L11-L24)
- [scatter](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/scatter.py#L6-L16)
- [scatterCI](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/plot/scatterCI.py#L7-L14)

#### cimcb_lite.cross_val
- [kfold](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/cross_val/kfold.pyy#L14-L42)

#### cimcb_lite.bootstrap
- [Perc](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/bootstrap/Perc.py#L6-L35)
- [BC](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/bootstrap/BC.py#L8-L37)
- [BCA](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/bootstrap/BCA.py#L8-L36)

#### cimcb_lite.utils
- [binary_metrics](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/binary_metrics.py#L5-L23)
- [ci95_ellipse](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/ci95_ellipse.py#L6-L28)
- [binary_metrics](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/binary_metrics.py#L5-L23)
- [knnimpute](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/knnimpute.py#L7-L22)
- [load_dataXL](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/load_dataXL.py#L7-L29)
- [nested_getattr](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/nested_getattr.py#L4-L5)
- [scale](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/scale.py#L4-L42)
- [table_check](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/table_check.py#L4-L17)
- [univariate_2class](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/univariate_2class.py#L8-L35)
- [wmean](https://github.com/KevinMMendez/cimcb_lite/blob/master/cimcb_lite/utils/wmean.py#L4-L19)

### License
cimcb_lite is licensed under the ___ license. 

### Authors
- Kevin Mendez
- [David Broadhurst](https://scholar.google.ca/citations?user=M3_zZwUAAAAJ&hl=en)

### Correspondence
Professor David Broadhurst, Director of the Centre for Integrative Metabolomics & Computation Biology at Edith Cowan University.  
E-mail: d.broadhurst@ecu.edu.au

### Citation
If you would cite cimcb_lite in a scientific publication, you can use the following: ___
