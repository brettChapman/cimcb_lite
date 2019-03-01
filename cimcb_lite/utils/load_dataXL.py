import pandas as pd
import numpy as np
from os import path
from .table_check import table_check


def load_dataXL(filename, DataSheet, PeakSheet):
    """Loads and validates the DataFile and PeakFile."""

    if path.isfile(filename) is False:
        raise ValueError("{} does not exist.".format(filename))

    if not filename.endswith('.xlsx'):
        raise ValueError("{} should be a .xlsx file.".format(filename))

    # LOAD PEAK DATA
    print("Loadings PeakFile: {}".format(PeakSheet))
    PeakTable = pd.read_excel(filename, sheet_name=PeakSheet)

    # LOAD DATA TABLE
    print("Loadings DataFile: {}".format(DataSheet))
    DataTable = pd.read_excel(filename, sheet_name=DataSheet)

    # Replace with nans
    DataTable = DataTable.replace(-99, np.nan)
    DataTable = DataTable.replace('.', np.nan)
    DataTable = DataTable.replace(' ', np.nan)
    
    # Error checks
    table_check(DataTable, PeakTable, print_statement=True)

    print("TOTAL SAMPLES: {} TOTAL PEAKS: {}".format(len(DataTable), len(PeakTable)))
    print("Done!")

    return DataTable, PeakTable
