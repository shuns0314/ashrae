
import datetime
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

DATA_PATH = "../m/ashrae-energy-prediction/"


def create_lgb_models(features, target, splits_num=3):
    # 各カテゴリが同じになるようにtrainとtestの分割する。
    kf = KFold(n_splits=splits_num)

    categorical_features = [
        "building_id",
        "site_id",
        "meter",
        "primary_use",
        "weekend"
    ]

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 1280,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse",
    }

    for fold_num, (train_index, test_index) in enumerate(kf.split(features)):
        train_features = features.loc[train_index]
        train_target = target.loc[train_index]

        test_features = features.loc[test_index]
        test_target = target.loc[test_index]

        d_training = lgb.Dataset(
            train_features,
            label=train_target,
            categorical_feature=categorical_features,
            free_raw_data=False)

        d_test = lgb.Dataset(
            test_features,
            label=test_target,
            categorical_feature=categorical_features,
            free_raw_data=False)

        model = lgb.train(
            params,
            train_set=d_training,
            num_boost_round=1000,
            valid_sets=[d_training, d_test],
            verbose_eval=25,
            early_stopping_rounds=50)

        del train_features, train_target, test_features, test_target, d_training, d_test
        gc.collect()

        model_name = f"lgb_models_{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{fold_num}"
        model.save_model(filename=f"ashrae/models/{model_name}")


def main():
    

    gc.collect()
    create_lgb_models(features, target)
