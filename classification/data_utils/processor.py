"""
Module with functions to process raw data into classifiable form.

This code uses the Template method design pattern. The AbstractDataProcessor class
has a method `process` which is an example of an abstract template method definition.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from pandas import DataFrame


class DataLocator:
    """Just holds the dictionaries for finding the path to certain data"""
    # dictionary of inputs paths
    RAW_DATA_PATHS = {
        'ecoli': 'classification/resources/raw_data/ecoli/ecoli.data',
        'glass': 'classification/resources/raw_data/glass/glass.data',
        'abalone': 'classification/resources/raw_data/abalone/abalone.data'
    }

    # dictionary of output paths
    DATA_PATHS = {
        'ecoli': 'classification/resources/data/ecoli.csv',
        'glass': 'classification/resources/data/glass.csv',
        'abalone': 'classification/resources/data/abalone.csv',
        'winequality': 'classification/resources/data/winequality-red.csv',
        'algerian': 'classification/resources/data/Algerian_forest_fires_dataset_UPDATE.csv'
    }


class AbstractDataProcessor(DataLocator, ABC):
    """
    The Abstract Class defining the template method for processing data.
    """
    NAME = 'ABSTRACT'

    @classmethod
    def process(cls) -> DataFrame:
        """
        This method defines the skeleton of the process method from raw data to
        correctly formatter csv file
        """
        name, df = cls.convert_raw_data_to_df()
        cls.save_as_csv(name, df)
        return df

    # The save functionality is the same for all data
    @classmethod
    def save_as_csv(cls, name: str, df: DataFrame) -> None:
        df.to_csv(cls.DATA_PATHS[name])

    # This operation has to be implemented in a subclass.
    @classmethod
    @abstractmethod
    def convert_raw_data_to_df(cls) -> Tuple[str, DataFrame]:
        """
        The child classes must implement this. The dataframe must have column
        names feature_i for all of the feature columns and a column of
        labels with the name label.

        This method may also transform the raw data features into a
        form more useful for classification such as one hot encoding.
        """
        pass

    @classmethod
    def get_raw_path(cls):
        return cls.RAW_DATA_PATHS[cls.NAME]

    @classmethod
    def get_path(cls):
        return cls.DATA_PATHS[cls.NAME]


class EcoliDataProcessor(AbstractDataProcessor):
    NAME = 'ecoli'

    @classmethod
    def convert_raw_data_to_df(cls) -> Tuple[str, DataFrame]:
        """Converts raw ecoli data to a dataframe"""
        # path to raw data file
        path = cls.get_raw_path()

        # names of columns in the dataframe
        names = ['sequence'] + [f'feature_{i}' for i in range(1, 8)] + ['label']

        # create dataframe
        df = pd.read_csv(path, delim_whitespace=True, names=names)

        return cls.NAME, df


class GlassDataProcessor(AbstractDataProcessor):
    NAME = 'glass'

    @classmethod
    def convert_raw_data_to_df(cls) -> Tuple[str, DataFrame]:
        """Converts raw glass data to a dataframe."""
        path = cls.get_raw_path()

        # name of columns in the dataframe
        names = ['index']+[f'feature_{i}' for i in range(9)]+['label']

        # TODO should we drop the index column

        # create dataframe
        df = pd.read_csv(path, names=names)

        return cls.NAME, df


class LetterDataProcessor(AbstractDataProcessor):
    NAME = 'letter'
    NUM_FEATURES = 16

    @classmethod
    def convert_raw_data_to_df(cls) -> Tuple[str, DataFrame]:
        """Converts raw letter data into a dataframe"""
        path = cls.get_raw_path()

        # names of columns
        names = ['label']+[f'feature_{i}' for i in range(cls.NUM_FEATURES)]

        df = pd.read_csv(path, names=names)

        return cls.NAME, df


class OptdigitsDataProcessor(AbstractDataProcessor):
    NAME = 'optdigits'
    NUM_FEATURES = 64

    @classmethod
    def convert_raw_data_to_df(cls) -> Tuple[str, DataFrame]:
        """converts raw optdigits data into dataframe"""
        path = cls.get_raw_path()

        # names of columns
        names = [f'feature_{i}' for i in range(cls.NUM_FEATURES)]+['label']

        df = pd.read_csv(path, names=names)

        return cls.NAME, df


class AbaloneDataProcessor(AbstractDataProcessor):
    NAME = 'abalone'
    NUM_FEATURES = 8

    @classmethod
    def convert_raw_data_to_df(cls) -> Tuple[str, DataFrame]:
        """converts raw optdigits data into dataframe"""
        path = cls.get_raw_path()

        # names of columns
        names = [f'feature_{i}' for i in range(cls.NUM_FEATURES)]+['label']

        df = pd.read_csv(path, names=names)

        return cls.NAME, df


# collect data processors in map for use my loaders
PROCESSOR_MAP = {
    'ecoli': EcoliDataProcessor,
    'glass': GlassDataProcessor,
    'letter': LetterDataProcessor,
    'optdigits': OptdigitsDataProcessor,
    'abalone': AbaloneDataProcessor
}
