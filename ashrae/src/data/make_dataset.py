# -*- coding: utf-8 -*-
"""Make datasets."""
import gc
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np

from ..utiles import reduce_mem_usage
from ..features.build_features import (
    features_engineering,
    fill_weather_dataset)


RAW_PATH = "../data/raw"
PROCESSED_PATH = "../data/processed"


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read csv
    train_df = pd.read_csv(RAW_PATH + '/train.csv')
    train_df.query("building_id == 1099").meter_reading.plot()
    train_df = train_df.query(
        'not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")'
        )
    building_df = pd.read_csv(RAW_PATH + '/building_metadata.csv')
    weather_df = pd.read_csv(RAW_PATH + '/weather_train.csv')

    # fix weather dataframe
    weather_df = fill_weather_dataset(weather_df)

    train_df = reduce_mem_usage(train_df, use_float16=True)
    building_df = reduce_mem_usage(building_df, use_float16=True)
    weather_df = reduce_mem_usage(weather_df, use_float16=True)

    train_df = train_df.merge(
        building_df,
        left_on='building_id',
        right_on='building_id',
        how='left')
    train_df = train_df.merge(
        weather_df,
        how='left',
        left_on=['site_id', 'timestamp'],
        right_on=['site_id', 'timestamp'])

    del weather_df
    gc.collect()

    train_df = features_engineering(train_df)
    target = pd.DataFrame(data=np.log1p(train_df["meter_reading"],
                          columns=["meter_reading"]))
    features = train_df.drop("meter_reading", axis=1)

    features.save("features.csv")
    target.save("target.csv")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
