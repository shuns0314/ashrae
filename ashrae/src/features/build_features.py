
import datetime
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


def fill_weather_dataset(weather_df):

    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(
        weather_df['timestamp'].min(), time_format)
    end_date = datetime.datetime.strptime(
        weather_df['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []

    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df, new_rows])
        weather_df = weather_df.reset_index(drop=True)

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id', 'day', 'month'])

    air_temperature_filler = pd.DataFrame(
        data=weather_df.groupby(['site_id', 'day', 'month'])['air_temperature'].mean(),
        columns=["air_temperature"])
    weather_df.update(air_temperature_filler, overwrite=False)

    # add lag feature
    air_temperature_filler = air_temperature_filler.reset_index().sort_values(by=["month", "day"])
    for site_id in range(16):
        index = air_temperature_filler.query(f"site_id == {site_id}").index
        air_temperature_filler.loc[
            index,
            "yesterday_air_temperature"
        ] = air_temperature_filler[air_temperature_filler.site_id==site_id]["air_temperature"].shift().values

    weather_df["yesterday_air_temperature"] = air_temperature_filler.set_index(
            ['site_id', 'day', 'month'])["yesterday_air_temperature"]

    # 前日との温度差
    weather_df["temperature_1day_gap"] = weather_df["yesterday_air_temperature"] - weather_df["air_temperature"]
    weather_df = mean_fillna(df=weather_df, column="cloud_coverage")
    weather_df = mean_fillna(df=weather_df, column="dew_temperature")

    # 露点温度と相対温度の差（湿数）: 大気の安定性を判別するための重要な気象要素らしい
    weather_df["dew_point_spread"] = weather_df["air_temperature"] - weather_df["dew_temperature"]
    weather_df = mean_fillna(df=weather_df, column="sea_level_pressure")

    # wind_directionの平均を取ると変なことになるので、範囲ごとにラベルづけして、最頻値をとる。
    weather_df.wind_direction = weather_df.wind_direction.apply(lambda x: direction_(x))

    wind_direction_filler = pd.DataFrame(
        data=weather_df.groupby(['site_id', 'day', 'month'])['wind_direction'].mode(),
        columns=['wind_direction'])
    weather_df.update(wind_direction_filler, overwrite=False)

    weather_df = mean_fillna(df=weather_df, column='wind_speed')

    # sin cos になおすとか
    # 風速成分をsin cosにかけちゃう
    # wind_directionが360だけど、wind_speedが0のやつは、wind_directionを0にそろえる。
    index = weather_df[["wind_direction", "wind_speed"]].query("wind_speed == 0 & wind_direction != 0").index
    weather_df.loc[index, "wind_direction"] = 0

    weather_df["precip_depth_1_hr_isna"] = weather_df.precip_depth_1_hr.isna().apply(lambda x: 1 if x is True else 0)

    precip_depth_isna_filler = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr_isna'].sum()
    weather_df["precip_depth_1_hr_isna_sum"] = precip_depth_isna_filler

    weather_df = mean_fillna(df=weather_df, column='precip_depth_1_hr')
    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)

    return weather_df


def mean_fillna(df, column):
    # Step 1
    filler = df.groupby(['site_id', 'day', 'month'])[column].mean()
    # Step 2
    filler = pd.DataFrame(
        data=filler.fillna(method='ffill'),
        columns=[column])
    df.update(filler, overwrite=False)
    return df


def features_engineering(df):
    # Sort by timestamp
    df.sort_values("timestamp")
    df.reset_index(drop=True)

    # Add more features
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    df['square_feet'] = np.log1p(df['square_feet'])

    # Remove Unused Columns
    drop = ["sea_level_pressure",
            "year_built",
            "floor_count"]
    df = df.drop(drop, axis=1)
    gc.collect()

    # Encode Categorical Data
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])

    return df

def direction_(direction):
    direct = np.nan
    if 0 <= direction and direction < 45:
        direct = 0
    elif 45 <= direction and direction < 90:
        direct = 1
    elif 90 <= direction and direction < 135:
        direct = 2
    elif 135 <= direction and direction < 180:
        direct = 3
    elif 180 <= direction and direction < 225:
        direct = 4
    elif 225 <= direction and direction < 270:
        direct = 5
    elif 270 <= direction:
        direct = 6
    return direct
