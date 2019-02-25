#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/20/19 4:24 PM
# @Author  : xiaowa

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def get_today_str():
    return datetime.strftime(datetime.today(), '%Y%m%d')


class OneHotEnc():
    def __init__(self):
        self.obe_dict = {}
        self.series_list = []
        self.ohe = None

    def fit(self, df, feature_list):
        for feature in feature_list:
            ser = df[feature]
            if df[feature].dtype == np.object:
                lbe = LabelEncoder()
                lbe.fit(ser)
                self.obe_dict[feature] = lbe
                ser = pd.Series(lbe.transform(ser), name=ser.name)
            self.series_list.append(ser)
        tmp_df = pd.concat(self.series_list, axis=1)
        self.ohe = OneHotEncoder()
        self.ohe.fit(tmp_df)
        return self.ohe

    def transform(self, df, feature_list):
        tmp = df[feature_list]
        for k, v in self.obe_dict.items():
            if k in feature_list:
                tmp[k] = v.transform(tmp[k])

        rs = self.ohe.transform(tmp)
        return rs.toarray()
