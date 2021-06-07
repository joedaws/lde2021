"""
Functions and classes for loading the data
"""
import os

import pandas as pd

from classification.data_utils.processor import DataLocator
from classification.data_utils.processor import PROCESSOR_MAP


def load_df(name) -> pd.DataFrame:
    """Loads a prepared dataframe.

    If the file has already been processed then we just load
    directly from the saved csv, otherwise, the dataprocessor instance is used.
    """
    if os.path.isfile(DataLocator.DATA_PATHS[name]):
        df = pd.read_csv(DataLocator.DATA_PATHS[name])
    else:
        # obtain data processor for this kind of data
        data_processor = PROCESSOR_MAP.get(name)
        # use processor to obtain dataframe (also saves csv to file)
        df = data_processor.process()

    return df
